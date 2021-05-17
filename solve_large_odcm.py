"""Compute a large Origin Destination (OD) cost matrices by chunking the
inputs, solving in parallel, and recombining the results into a single
feature class.

This is a sample script users can modify to fit their specific needs.

This script can be called from the script tool definition or from the command line.

Copyright 2021 Esri
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import sys
import time
import traceback
import argparse
import subprocess
from distutils.util import strtobool

import arcpy

import helpers
from od_config import OD_PROPS  # Import OD Cost Matrix settings from config file

arcpy.env.overwriteOutput = True


class ODCostMatrixSolver():  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Compute OD Cost Matrices between Origins and Destinations in parallel and combine results.

    This class preprocesses and validate inputs and then spins up a subprocess to do the actual OD Cost Matrix
    calculations. This is necessary because the a script tool running in the ArcGIS Pro UI cannot directly call
    multiprocessing using concurrent.futures. We must spin up a subprocess, and the subprocess must spawn parallel
    processes for the calculations. Thus, this class does all the pre-processing, passes inputs to the subprocess, and
    handles messages returned by the subprocess. The subprocess actually does the calculations.
    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, origins, destinations, network_data_source, travel_mode, output_od_lines, output_origins,
        output_destinations, chunk_size, max_processes, time_units, distance_units, cutoff=None, num_destinations=None,
        precalculate_network_locations=True, barriers=None
    ):
        """Initialize the ODCostMatrixSolver class.

        Args:
            origins (str, layer): Catalog path or layer for the input origins
            destinations (str, layer): Catalog path or layer for the input destinations
            network_data_source (str, layer): Catalog path, layer, or URL for the input network dataset
            travel_mode (str, travel mode): Travel mode object, name, or json string representation
            output_od_lines (str): Catalog path to the output OD Lines feature class
            output_origins (str): Catalog path to the output Origins feature class
            output_destinations (str): Catalog path to the output Destinations feature class
            chunk_size (int): Maximum number of origins and destinations that can be in one chunk
            max_processes (int): Maximum number of allowed parallel processes
            time_units (str): String representation of time units
            distance_units (str): String representation of distance units
            cutoff (float, optional): Impedance cutoff to limit the OD Cost Matrix solve. Interpreted in the time_units
                if the travel mode is time-based. Interpreted in the distance-units if the travel mode is distance-
                based. Interpreted in the impedance units if the travel mode is neither time- nor distance-based.
                Defaults to None. When None, do not use a cutoff.
            num_destinations (int, optional): The number of destinations to find for each origin. Defaults to None,
                which means to find all destinations.
            precalculate_network_locations (bool, optional): Whether to precalculate network location fields for all
                inputs. Defaults to True. Should be false if the network_data_source is a service.
            barriers (list(str, layer), optional): List of catalog paths or layers for point, line, and polygon barriers
                 to use. Defaults to None.
        """
        self.origins = origins
        self.destinations = destinations
        self.network_data_source = network_data_source
        self.travel_mode = travel_mode
        self.output_od_lines = output_od_lines
        self.output_origins = output_origins
        self.output_destinations = output_destinations
        self.chunk_size = chunk_size
        self.max_processes = max_processes
        self.time_units = time_units
        self.distance_units = distance_units
        self.cutoff = cutoff
        self.num_destinations = num_destinations
        self.should_precalc_network_locations = precalculate_network_locations
        self.barriers = barriers if barriers else []

        self.same_origins_destinations = bool(self.origins == self.destinations)

        self.max_origins = self.chunk_size
        self.max_destinations = self.chunk_size

        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.service_limits = None
        self.is_agol = False

    def _validate_inputs(self):
        """Validate the OD Cost Matrix inputs."""
        # Validate input numerical values
        if self.chunk_size < 1:
            err = "Chunk size must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.max_processes < 1:
            err = "Maximum allowed parallel processes must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.cutoff not in ["", None] and self.cutoff <= 0:
            err = "Impedance cutoff must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.num_destinations not in ["", None] and self.num_destinations < 1:
            err = "Number of destinations to find must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)

        # Validate origins, destinations, and barriers
        self._validate_input_feature_class(self.origins)
        self._validate_input_feature_class(self.destinations)
        for barrier_fc in self.barriers:
            self._validate_input_feature_class(barrier_fc)

        # Validate network
        if not self.is_service and not arcpy.Exists(self.network_data_source):
            err = f"Input network dataset {self.network_data_source} does not exist."
            arcpy.AddError(err)
            raise ValueError(err)
        if not self.is_service:
            # Try to check out the Network Analyst extension
            try:
                arcpy.CheckOutExtension("network")
            except Exception as ex:
                err = "Unable to check out Network Analyst extension license."
                arcpy.AddError(err)
                raise RuntimeError(err) from ex
            # If the network dataset is a layer, convert it to a catalog path so we can pass it to the subprocess
            if hasattr(self.network_data_source, "dataSource"):
                self.network_data_source = self.network_data_source.dataSource

        # Validate OD Cost Matrix settings and convert travel mode to a JSON string
        self.travel_mode = self._validate_od_settings()

        # For a services solve, get tool limits and validate max processes and chunk size
        if self.is_service:
            self._get_tool_limits_and_is_agol()
            if self.is_agol and self.max_processes > helpers.MAX_AGOL_PROCESSES:
                arcpy.AddWarning((
                    f"The specified maximum number of parallel processes, {self.max_processes}, exceeds the limit of "
                    f"{helpers.MAX_AGOL_PROCESSES} allowed when using as the network data source the ArcGIS Online "
                    "services or a hybrid portal whose network analysis services fall back to the ArcGIS Online "
                    "services. The maximum number of parallel processes has been reduced to "
                    f"{helpers.MAX_AGOL_PROCESSES}."))
                self.max_processes = helpers.MAX_AGOL_PROCESSES
            self._update_max_inputs_for_service()
            if self.should_precalc_network_locations:
                arcpy.AddWarning(
                    "Cannot precalculate network location fields when the network data source is a service.")
                self.should_precalc_network_locations = False

    @staticmethod
    def _validate_input_feature_class(feature_class):
        """Validate that the designated input feature class exists and is not empty.

        Args:
            feature_class (str, layer): Input feature class or layer to validate

        Raises:
            ValueError: The input feature class does not exist.
            ValueError: The input feature class has no rows.
        """
        if not arcpy.Exists(feature_class):
            err = f"Input dataset {feature_class} does not exist."
            arcpy.AddError(err)
            raise ValueError(err)
        if int(arcpy.management.GetCount(feature_class).getOutput(0)) <= 0:
            err = f"Input dataset {feature_class} has no rows."
            arcpy.AddError(err)
            raise ValueError(err)

    def _validate_od_settings(self):
        """Validate OD cost matrix settings by spinning up a dummy OD Cost Matrix object.

        Raises:
            ValueError: If the travel mode doesn't have a name

        Returns:
            str: JSON string representation of the travel mode
        """
        arcpy.AddMessage("Validating OD Cost Matrix settings...")
        # Validate time and distance units
        time_units = helpers.convert_time_units_str_to_enum(self.time_units)
        distance_units = helpers.convert_distance_units_str_to_enum(self.distance_units)
        # Create a dummy ODCostMatrix object, initialize an OD solver object, and set properties
        try:
            odcm = arcpy.nax.OriginDestinationCostMatrix(self.network_data_source)
            odcm.travelMode = self.travel_mode
            odcm.timeUnits = time_units
            odcm.distanceUnits = distance_units
            odcm.defaultImpedanceCutoff = self.cutoff
            odcm.defaultDestinationCount = self.num_destinations
        except Exception:
            arcpy.AddError("Invalid OD Cost Matrix settings.")
            errs = traceback.format_exc().splitlines()
            for err in errs:
                arcpy.AddError(err)
            raise

        # Return a JSON string representation of the travel mode to pass to the subprocess
        return odcm.travelMode._JSON

    def _get_tool_limits_and_is_agol(
            self, service_name="asyncODCostMatrix", tool_name="GenerateOriginDestinationCostMatrix"):
        """Retrieve a dictionary of various limits supported by a portal tool and whether the portal uses AGOL services.

        Assumes that we have already determined that the network data source is a service.

        Args:
            service_name (str, optional): Name of the service. Defaults to "asyncODCostMatrix".
            tool_name (str, optional): Tool name for the designated service. Defaults to
                "GenerateOriginDestinationCostMatrix".
        """
        arcpy.AddMessage("Getting tool limits from the portal...")
        if not self.network_data_source.endswith("/"):
            self.network_data_source = self.network_data_source + "/"
        try:
            tool_info = arcpy.nax.GetWebToolInfo(service_name, tool_name, self.network_data_source)
            # serviceLimits returns the maximum origins and destinations allowed by the service, among other things
            self.service_limits = tool_info["serviceLimits"]
            # isPortal returns True for Enterprise portals and False for AGOL or hybrid portals that fall back to using
            # the AGOL services
            self.is_agol = not tool_info["isPortal"]
        except Exception:
            arcpy.AddError("Error getting tool limits from the portal.")
            errs = traceback.format_exc().splitlines()
            for err in errs:
                arcpy.AddError(err)
            raise

    def _update_max_inputs_for_service(self):
        """Check the user's specified max origins and destinations and reduce max to portal limits if required."""
        lim_max_origins = int(self.service_limits["maximumOrigins"])
        if lim_max_origins < self.max_origins:
            self.max_origins = lim_max_origins
            arcpy.AddMessage(
                f"Max origins per chunk has been updated to {self.max_origins} to accommodate service limits.")
        lim_max_destinations = int(self.service_limits["maximumDestinations"])
        if lim_max_destinations < self.max_destinations:
            self.max_destinations = lim_max_destinations
            arcpy.AddMessage(
                f"Max destinations per chunk has been updated to {self.max_destinations} to accommodate service limits."
            )

    @staticmethod
    def _spatially_sort_input(input_features, tracked_oid_name):
        """Spatially sort the input feature class.

        Also adds a field to the input feature class to preserve the original OID values. This field is called
        "OriginOID" for origins and "DestinationOID" for destinations.

        Args:
            input_features (str): Catalog path to the feature class to sort
            tracked_oid_name (str): New field name to store original OIDs.
        """
        arcpy.AddMessage(f"Spatially sorting input dataset {input_features}...")

        # Add a unique ID field so we don't lose OID info when we sort and can use these later in joins.
        # Note that if the original input was a shapefile, these IDs will likely be wrong because copying the original
        # input to the output geodatabase will have altered the original ObjectIDs.
        # Consequently, don't use shapefiles as inputs.
        desc = arcpy.Describe(input_features)
        if tracked_oid_name in [f.name for f in desc.fields]:
            arcpy.management.DeleteField(input_features, tracked_oid_name)
        arcpy.management.AddField(input_features, tracked_oid_name, "LONG")
        arcpy.management.CalculateField(input_features, tracked_oid_name, f"!{desc.oidFieldName}!")

        # Make a temporary copy of the inputs so the Sort tool can write its output to the input_features path, which is
        # the ultimate desired location
        temp_inputs = arcpy.CreateUniqueName("TempODInputs", arcpy.env.scratchGDB)  # pylint:disable = no-member
        arcpy.management.Copy(input_features, temp_inputs)

        # Spatially sort input features
        try:
            arcpy.management.Sort(temp_inputs, input_features, [[desc.shapeFieldName, "ASCENDING"]], "PEANO")
        except arcpy.ExecuteError:  # pylint:disable = no-member
            msgs = arcpy.GetMessages(2)
            if "000824" in msgs:  # ERROR 000824: The tool is not licensed.
                arcpy.AddWarning("Skipping spatial sorting because the Advanced license is not available.")
            else:
                arcpy.AddWarning(f"Skipping spatial sorting because the tool failed. Messages:\n{msgs}")

        # Clean up. Delete temporary copy of inputs
        arcpy.management.Delete([temp_inputs])

    def _precalculate_network_locations(self, input_features):
        """Precalculate network location fields if possible for faster loading and solving later.

        Cannot be used if the network data source is a service. Uses the searchTolerance, searchToleranceUnits, and
        searchQuery properties set in the OD config file.

        Args:
            input_features (feature class catalog path): Feature class to calculate network locations for
            network_data_source (network dataset catalog path): Network dataset to use to calculate locations
            travel_mode (travel mode): Travel mode name, object, or json representation to use when calculating
            locations.
        """
        if self.is_service:
            arcpy.AddMessage(
                "Skipping precalculating network location fields because the network data source is a service.")
            return

        arcpy.AddMessage(f"Precalculating network location fields for {input_features}...")

        # Get location settings from config file if present
        search_tolerance = None
        if "searchTolerance" in OD_PROPS and "searchToleranceUnits" in OD_PROPS:
            search_tolerance = f"{OD_PROPS['searchTolerance']} {OD_PROPS['searchToleranceUnits'].name}"
        search_query = OD_PROPS.get("search_query", None)

        # Calculate network location fields if network data source is local
        arcpy.na.CalculateLocations(
            input_features, self.network_data_source,
            search_tolerance=search_tolerance,
            search_query=search_query,
            travel_mode=self.travel_mode
        )

    def _preprocess_inputs(self):
        """Preprocess the input feature classes to prepare them for use in the OD Cost Matrix."""
        # Copy Origins and Destinations to outputs
        arcpy.AddMessage("Copying input origins and destinations to outputs...")
        arcpy.management.Copy(self.origins, self.output_origins)
        if not self.same_origins_destinations:
            arcpy.management.Copy(self.destinations, self.output_destinations)

        # Spatially sort inputs
        tracked_origin_oid = "OriginOID"
        tracked_destination_oid = "DestinationOID"
        self._spatially_sort_input(self.output_origins, tracked_origin_oid)
        if not self.same_origins_destinations:
            self._spatially_sort_input(self.output_destinations, tracked_destination_oid)

        # Precalculate network location fields for inputs
        if not self.is_service and self.should_precalc_network_locations:
            self._precalculate_network_locations(self.output_origins)
            if not self.same_origins_destinations:
                self._precalculate_network_locations(self.output_destinations)
            for barrier_fc in self.barriers:
                self._precalculate_network_locations(barrier_fc)

        # If Origins and Destinations were the same, copy the output origins to the output destinations. This saves us
        # from having to spatially sort and precalculate network locations on the same feature class twice.
        if self.same_origins_destinations:
            arcpy.management.Copy(self.output_origins, self.output_destinations)
            # Update the tracked OID field name
            arcpy.management.AlterField(
                self.output_destinations,
                tracked_origin_oid,
                tracked_destination_oid,
                tracked_destination_oid
            )

    def _execute_solve(self):
        """Solve the OD Cost Matrix analysis."""
        # Launch the parallel_odcm script as a subprocess so it can spawn parallel processes. We have to do this because
        # a tool running in the Pro UI cannot call concurrent.futures without opening multiple instances of Pro.
        cwd = os.path.dirname(os.path.abspath(__file__))
        odcm_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(cwd, "parallel_odcm.py"),
            "--origins", self.output_origins,
            "--destinations", self.output_destinations,
            "--output-od-lines", self.output_od_lines,
            "--network-data-source", self.network_data_source,
            "--travel-mode", self.travel_mode,
            "--time-units", self.time_units,
            "--distance-units", self.distance_units,
            "--max-origins", str(self.max_origins),
            "--max-destinations", str(self.max_destinations),
            "--max-processes", str(self.max_processes),
        ]
        if self.barriers:
            odcm_inputs += ["--barriers"]
            odcm_inputs += self.barriers
        if self.cutoff:
            odcm_inputs += ["--cutoff", str(self.cutoff)]
        if self.num_destinations:
            odcm_inputs += ["--num-destinations", str(self.num_destinations)]
        # We do not want to show the console window when calling the command line tool from within our GP tool.
        # This can be done by setting this hex code.
        create_no_window = 0x08000000
        with subprocess.Popen(
            odcm_inputs,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=create_no_window
        ) as process:
            # The while loop reads the subprocess's stdout in real time and writes the stdout messages to the GP UI.
            # This is the only way to write the subprocess's status messages in a way that a user running the tool from
            # the ArcGIS Pro UI can actually see them.
            # When process.poll() returns anything other than None, the process has completed, and we should stop
            # checking and move on.
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    msg_string = output.strip().decode()
                    helpers.parse_std_and_write_to_gp_ui(msg_string)
                time.sleep(.5)

            # Once the process is finished, check if any additional errors were returned. Messages that came after the
            # last process.poll() above will still be in the queue here. This is especially important for detecting
            # messages from raised exceptions, especially those with tracebacks.
            output, _ = process.communicate()
            if output:
                out_msgs = output.decode().splitlines()
                for msg in out_msgs:
                    helpers.parse_std_and_write_to_gp_ui(msg)

            # In case something truly horrendous happened and none of the logging caught our errors, at least fail the
            # tool when the subprocess returns an error code. That way the tool at least doesn't happily succeed but not
            # actually do anything.
            return_code = process.returncode
            if return_code != 0:
                arcpy.AddError("OD Cost Matrix script failed.")

    def solve_large_od_cost_matrix(self):
        """Solve the large OD Cost Matrix in parallel."""
        try:
            self._validate_inputs()
            arcpy.AddMessage("Inputs successfully validated.")
        except Exception:  # pylint: disable=broad-except
            arcpy.AddError("Invalid inputs.")
            return

        # Preprocess inputs
        self._preprocess_inputs()

        # Solve the analysis
        self._execute_solve()


def _run_from_command_line():
    """Read arguments from the command line and run the tool."""
    # Create the parser
    parser = argparse.ArgumentParser(description=globals().get("__doc__", ""), fromfile_prefix_chars='@')

    # Define Arguments supported by the command line utility

    # --origins parameter
    help_string = "The full catalog path to the feature class containing the origins."
    parser.add_argument("-o", "--origins", action="store", dest="origins", help=help_string, required=True)

    # --destinations parameter
    help_string = "The full catalog path to the feature class containing the destinations."
    parser.add_argument("-d", "--destinations", action="store", dest="destinations", help=help_string, required=True)

    # --output-od-lines parameter
    help_string = "The catalog path to the output feature class that will contain the combined OD Cost Matrix results."
    parser.add_argument(
        "-ol", "--output-od-lines", action="store", dest="output_od_lines", help=help_string, required=True)

    # --output-origins parameter
    help_string = "The catalog path to the output feature class that will contain the updated origins."
    parser.add_argument(
        "-oo", "--output-origins", action="store", dest="output_origins", help=help_string, required=True)

    # --output-destinations parameter
    help_string = "The catalog path to the output feature class that will contain the updated destinations."
    parser.add_argument(
        "-od", "--output-destinations", action="store", dest="output_destinations", help=help_string, required=True)

    # --network-data-source parameter
    help_string = "The full catalog path to the network dataset or a portal url that will be used for the analysis."
    parser.add_argument(
        "-n", "--network-data-source", action="store", dest="network_data_source", help=help_string, required=True)

    # --travel-mode parameter
    help_string = (
        "A JSON string representation or string name of a travel mode from the network data source that will be used "
        "for the analysis."
    )
    parser.add_argument("-tm", "--travel-mode", action="store", dest="travel_mode", help=help_string, required=True)

    # --time-units parameter
    help_string = "String name of the time units for the analysis. These units will be used in the output."
    parser.add_argument("-tu", "--time-units", action="store", dest="time_units", help=help_string, required=True)

    # --distance-units parameter
    help_string = "String name of the distance units for the analysis. These units will be used in the output."
    parser.add_argument(
        "-du", "--distance-units", action="store", dest="distance_units", help=help_string, required=True)

    # --chunk-size parameter
    help_string = (
        "Maximum number of origins and destinations that can be in one chunk for parallel processing of OD Cost Matrix "
        "solves. For example, 1000 means that a chunk consists of no more than 1000 origins and 1000 destinations."
    )
    parser.add_argument(
        "-ch", "--chunk-size", action="store", dest="chunk_size", type=int, help=help_string, required=True)

    # --max-processes parameter
    help_string = "Maximum number parallel processes to use for the OD Cost Matrix solves."
    parser.add_argument(
        "-mp", "--max-processes", action="store", dest="max_processes", type=int, help=help_string, required=True)

    # --cutoff parameter
    help_string = (
        "Impedance cutoff to limit the OD cost matrix search distance. Should be specified in the same units as the "
        "time-units parameter if the travel mode's impedance is in units of time or in the same units as the "
        "distance-units parameter if the travel mode's impedance is in units of distance. Otherwise, specify this in "
        "the units of the travel mode's impedance attribute."
    )
    parser.add_argument(
        "-co", "--cutoff", action="store", dest="cutoff", type=float, help=help_string, required=False)

    # --num-destinations parameter
    help_string = "The number of destinations to find for each origin. Set to None to find all destinations."
    parser.add_argument(
        "-nd", "--num-destinations", action="store", dest="num_destinations", type=int, help=help_string,
        required=False)

    # --precalculate-network-locations parameter
    help_string = "Whether or not to precalculate network location fields before solving the OD Cost  Matrix."
    parser.add_argument(
        "-pnl", "--precalculate-network-locations", action="store", type=lambda x: bool(strtobool(x)),
        dest="precalculate_network_locations", help=help_string, required=True)

    # --barriers parameter
    help_string = "A list of catalog paths to the feature classes containing barriers to use in the OD Cost Matrix."
    parser.add_argument(
        "-b", "--barriers", action="store", dest="barriers", help=help_string, nargs='*', required=False)

    # Get arguments as dictionary.
    args = vars(parser.parse_args())

    # Solve the OD Cost Matrix
    od_solver = ODCostMatrixSolver(**args)
    od_solver.solve_large_od_cost_matrix()


if __name__ == "__main__":
    # Run script from the command line
    _run_from_command_line()
