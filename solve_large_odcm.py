"""Compute a large Origin Destination (OD) cost matrix by chunking the
inputs and solving in parallel. Write outputs into a single combined
feature class, a collection of CSV files, or a collection of Apache
Arrow files.

This is a sample script users can modify to fit their specific needs.

This script can be called from the script tool definition or from the command line.

Copyright 2023 Esri
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
import datetime
import traceback
import argparse
import subprocess
from distutils.util import strtobool

import arcpy

import helpers
from od_config import OD_PROPS, OD_PROPS_SET_BY_TOOL  # Import OD Cost Matrix settings from config file

arcpy.env.overwriteOutput = True


class ODCostMatrixSolver(
    helpers.PrecalculateLocationsMixin
):  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Compute OD Cost Matrices between Origins and Destinations in parallel and combine results.

    This class preprocesses and validates inputs and then spins up a subprocess to do the actual OD Cost Matrix
    calculations. This is necessary because the a script tool running in the ArcGIS Pro UI cannot directly call
    multiprocessing using concurrent.futures. We must spin up a subprocess, and the subprocess must spawn parallel
    processes for the calculations. Thus, this class does all the pre-processing, passes inputs to the subprocess, and
    handles messages returned by the subprocess. The subprocess, parallel_odcm.py, actually does the calculations.
    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, origins, destinations, network_data_source, travel_mode, output_origins,
        output_destinations, chunk_size, max_processes, time_units, distance_units, output_format,
        output_od_lines=None, output_data_folder=None, cutoff=None, num_destinations=None, time_of_day=None,
        precalculate_network_locations=True, sort_inputs=True, barriers=None
    ):
        """Initialize the ODCostMatrixSolver class.

        Args:
            origins (str, layer): Catalog path or layer for the input origins
            destinations (str, layer): Catalog path or layer for the input destinations
            network_data_source (str, layer): Catalog path, layer, or URL for the input network dataset
            travel_mode (str, travel mode): Travel mode object, name, or json string representation
            output_origins (str): Catalog path to the output Origins feature class
            output_destinations (str): Catalog path to the output Destinations feature class
            chunk_size (int): Maximum number of origins and destinations that can be in one chunk
            max_processes (int): Maximum number of allowed parallel processes
            time_units (str): String representation of time units
            distance_units (str): String representation of distance units
            output_format (str): String representation of the output format
            output_od_lines (str, optional): Catalog path to the output OD Lines feature class. Required if
                output_format is "Feature class".
            output_data_folder (str, optional): Catalog path to the output folder where CSV or Arrow files will be
                stored. Required if output_format is "CSV files" or "Apache Arrow files".
            cutoff (float, optional): Impedance cutoff to limit the OD Cost Matrix solve. Interpreted in the time_units
                if the travel mode is time-based. Interpreted in the distance-units if the travel mode is distance-
                based. Interpreted in the impedance units if the travel mode is neither time- nor distance-based.
                Defaults to None. When None, do not use a cutoff.
            num_destinations (int, optional): The number of destinations to find for each origin. Defaults to None,
                which means to find all destinations.
            time_of_day (str): String representation of the start time for the analysis ("%Y%m%d %H:%M" format)
            precalculate_network_locations (bool, optional): Whether to precalculate network location fields for all
                inputs. Defaults to True. Should be false if the network_data_source is a service.
            sort_inputs (bool, optional): Whether to spatially sort origins and destinations. Defaults to True.
            barriers (list(str, layer), optional): List of catalog paths or layers for point, line, and polygon barriers
                 to use. Defaults to None.
        """
        self.origins = origins
        self.destinations = destinations
        self.network_data_source = network_data_source
        self.travel_mode = travel_mode
        self.chunk_size = chunk_size
        self.max_processes = max_processes
        self.time_units = time_units
        self.distance_units = distance_units
        self.cutoff = cutoff
        self.num_destinations = num_destinations
        self.time_of_day = time_of_day
        self.time_of_day_dt = None  # Set during validation
        self.should_precalc_network_locations = precalculate_network_locations
        self.sort_inputs = sort_inputs
        self.barriers = barriers if barriers else []

        self.output_origins = output_origins
        self.output_destinations = output_destinations
        self.output_format_str = output_format
        self.output_format = None  # Set during validation
        self.output_od_lines = output_od_lines
        self.output_data_folder = output_data_folder

        self.same_origins_destinations = helpers.are_input_layers_the_same(self.origins, self.destinations)

        self.max_origins = self.chunk_size
        self.max_destinations = self.chunk_size

        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.service_limits = None
        self.is_agol = False

    def _validate_inputs(self):
        """Validate the OD Cost Matrix inputs."""
        # Validate the output format and ensure proper output location has been specified.
        self.output_format = helpers.convert_output_format_str_to_enum(self.output_format_str)
        if self.output_format is helpers.OutputFormat.featureclass:
            if not self.output_od_lines:
                err = f"Output OD Lines Feature Class is required when the output format is {self.output_format_str}."
                arcpy.AddError(err)
                raise ValueError(err)
        else:
            if not self.output_data_folder:
                err = f"Output Folder is required when the output format is {self.output_format_str}."
                arcpy.AddError(err)
                raise ValueError(err)

        # Validate that if the output format is Arrow:
        # - The Pro version is >= 2.9. Arrow output was not supported in earlier versions of Pro.
        # - The network data source is not a service. Arrow output from services solves is not yet supported.
        if self.output_format is helpers.OutputFormat.arrow:
            if helpers.arcgis_version < "2.9":
                err = f"{self.output_format_str} output format is not available in versions of ArcGIS Pro prior to 2.9."
                arcpy.AddError(err)
                raise RuntimeError(err)
            if self.is_service:
                err = (f"{self.output_format_str} output format is not available when a service is used as the network "
                       "data source.")
                arcpy.AddError(err)
                raise ValueError(err)

        # Validate input numerical values
        if self.chunk_size < 1:
            err = "Chunk size must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.max_processes < 1:
            err = "Maximum allowed parallel processes must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.max_processes > helpers.MAX_ALLOWED_MAX_PROCESSES:
            err = (
                f"The maximum allowed parallel processes cannot exceed {helpers.MAX_ALLOWED_MAX_PROCESSES:} due "
                "to limitations imposed by Python's concurrent.futures module."
            )
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

        # Validate time of day
        if self.time_of_day:
            try:
                self.time_of_day_dt = datetime.datetime.strptime(self.time_of_day, helpers.DATETIME_FORMAT)
            except ValueError as ex:
                arcpy.AddError(f"Could not convert input time of day to datetime: {str(ex)}")
                raise ex

        # Validate origins, destinations, and barriers
        helpers.validate_input_feature_class(self.origins)
        helpers.validate_input_feature_class(self.destinations)
        for barrier_fc in self.barriers:
            helpers.validate_input_feature_class(barrier_fc)

        # Validate network
        self.network_data_source = helpers.validate_network_data_source(self.network_data_source)

        # Validate OD Cost Matrix settings and convert travel mode to a JSON string
        self.travel_mode = self._validate_od_settings()

        # For a services solve, get tool limits and validate max processes and chunk size
        if self.is_service:
            self.service_limits, self.is_agol = helpers.get_tool_limits_and_is_agol(
                self.network_data_source, "asyncODCostMatrix", "GenerateOriginDestinationCostMatrix")
            if self.is_agol:
                self.max_processes = helpers.update_agol_max_processes(self.max_processes)
            self._update_max_inputs_for_service()
            if self.should_precalc_network_locations:
                arcpy.AddWarning(
                    "Cannot precalculate network location fields when the network data source is a service.")
                self.should_precalc_network_locations = False

        # Check licensing constraints
        if helpers.arc_license != "ArcInfo":
            if self.sort_inputs:
                arcpy.AddWarning("Cannot spatially sort inputs without the Advanced license.")
                self.sort_inputs = False

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
        # Create a dummy ODCostMatrix object and set properties
        try:
            odcm = arcpy.nax.OriginDestinationCostMatrix(self.network_data_source)
            odcm.travelMode = self.travel_mode
            odcm.timeUnits = time_units
            odcm.distanceUnits = distance_units
            odcm.defaultImpedanceCutoff = self.cutoff
            odcm.defaultDestinationCount = self.num_destinations
            odcm.timeOfDay = self.time_of_day_dt
        except Exception:
            arcpy.AddError("Invalid OD Cost Matrix settings.")
            errs = traceback.format_exc().splitlines()
            for err in errs:
                arcpy.AddError(err)
            raise
        # Set properties from config file
        for prop, value in OD_PROPS.items():
            if prop not in OD_PROPS_SET_BY_TOOL:
                try:
                    setattr(odcm, prop, value)
                except Exception:  # pylint: disable=broad-except
                    # Suppress errors for older services (pre 11.0) that don't support locate settings and services
                    # that don't support accumulating attributes because we don't want the tool to always fail.
                    if not (self.is_service and prop in [
                        "searchTolerance", "searchToleranceUnits", "accumulateAttributeNames"
                    ]):
                        err = f"Failed to set property {prop} from OD config file."
                        arcpy.AddError(err)
                        raise

        # Return a JSON string representation of the travel mode to pass to the subprocess
        return odcm.travelMode._JSON  # pylint: disable=protected-access

    def _update_max_inputs_for_service(self):
        """Check the user's specified max origins and destinations and reduce max to portal limits if required."""
        lim_max_origins = self.service_limits["maximumOrigins"]
        if lim_max_origins:
            lim_max_origins = int(lim_max_origins)
            if lim_max_origins < self.max_origins:
                self.max_origins = lim_max_origins
                arcpy.AddMessage(
                    f"Max origins per chunk has been updated to {self.max_origins} to accommodate service limits.")
        lim_max_destinations = self.service_limits["maximumDestinations"]
        if lim_max_destinations:
            lim_max_destinations = int(lim_max_destinations)
            if lim_max_destinations < self.max_destinations:
                self.max_destinations = lim_max_destinations
                arcpy.AddMessage((
                    f"Max destinations per chunk has been updated to {self.max_destinations} to accommodate service "
                    "limits."
                ))

    @staticmethod
    def _add_tracked_oid_field(input_features, tracked_oid_name):
        """Add a new field in the data and calculate it to equal the ObjectID.

        This is primarily used when sorting data so we don't lose the original ObjectID of the input. It is also useful
        in relating the output OD Cost Matrix results to the inputs since the relationship is more obvious when the
        fields are called OriginOID and DestinationOID.

        Note: In ArcGIS Pro 2.9, the Sort tool was enhanced to include an ORIG_FID field in the output tracking the
        original ObjectID. However, since we want this code sample to be compatible with older versions of ArcGIS Pro,
        this more complicated implementation of ObjectID field tracking has been maintained.
        Note that if the original input was a shapefile, these IDs will likely be wrong because copying the original
        input to the output geodatabase will have altered the original ObjectIDs.
        Consequently, don't use shapefiles as inputs.
        """
        desc = arcpy.Describe(input_features)
        if tracked_oid_name in [f.name for f in desc.fields]:
            arcpy.management.DeleteField(input_features, tracked_oid_name)
        arcpy.management.AddField(input_features, tracked_oid_name, "LONG")
        arcpy.management.CalculateField(input_features, tracked_oid_name, f"!{desc.oidFieldName}!")

    @staticmethod
    def _spatially_sort_input(input_features):
        """Spatially sort the input feature class.

        Also adds a field to the input feature class to preserve the original OID values. This field is called
        "OriginOID" for origins and "DestinationOID" for destinations.

        Args:
            input_features (str): Catalog path to the feature class to sort
            tracked_oid_name (str): New field name to store original OIDs.
        """
        arcpy.AddMessage(f"Spatially sorting input dataset {input_features}...")

        # Make a temporary copy of the inputs so the Sort tool can write its output to the input_features path, which is
        # the ultimate desired location
        temp_inputs = arcpy.CreateUniqueName("TempODInputs", arcpy.env.scratchGDB)  # pylint:disable = no-member
        arcpy.management.Copy(input_features, temp_inputs)

        # Spatially sort input features
        try:
            arcpy.management.Sort(
                temp_inputs, input_features, [[arcpy.Describe(input_features).shapeFieldName, "ASCENDING"]], "PEANO")
        except arcpy.ExecuteError:  # pylint:disable = no-member
            msgs = arcpy.GetMessages(2)
            if "000824" in msgs:  # ERROR 000824: The tool is not licensed.
                arcpy.AddWarning("Skipping spatial sorting because the Advanced license is not available.")
            else:
                arcpy.AddWarning(f"Skipping spatial sorting because the tool failed. Messages:\n{msgs}")

        # Clean up. Delete temporary copy of inputs
        arcpy.management.Delete([temp_inputs])

    def _preprocess_inputs(self):
        """Preprocess the input feature classes to prepare them for use in the OD Cost Matrix."""
        # Copy Origins and Destinations to outputs
        arcpy.AddMessage("Copying input origins and destinations to outputs...")
        arcpy.conversion.FeatureClassToFeatureClass(
            self.origins,
            os.path.dirname(self.output_origins),
            os.path.basename(self.output_origins)
        )
        if not self.same_origins_destinations:
            arcpy.conversion.FeatureClassToFeatureClass(
                self.destinations,
                os.path.dirname(self.output_destinations),
                os.path.basename(self.output_destinations)
            )

        # Preserve original OIDs
        tracked_origin_oid = "OriginOID"
        tracked_destination_oid = "DestinationOID"
        self._add_tracked_oid_field(self.output_origins, tracked_origin_oid)
        if not self.same_origins_destinations:
            self._add_tracked_oid_field(self.output_destinations, tracked_destination_oid)

        # Spatially sort inputs
        if self.sort_inputs:
            self._spatially_sort_input(self.output_origins)
            if not self.same_origins_destinations:
                self._spatially_sort_input(self.output_destinations)

        # Precalculate network location fields for inputs
        if not self.is_service and self.should_precalc_network_locations:
            self.output_origins = self._precalculate_locations(self.output_origins, OD_PROPS)
            if not self.same_origins_destinations:
                self.output_destinations = self._precalculate_locations(self.output_destinations, OD_PROPS)
            updated_barriers = []
            for barrier_fc in self.barriers:
                updated_barriers.append(self._precalculate_locations(barrier_fc, OD_PROPS))
            self.barriers = updated_barriers

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
        odcm_inputs = [
            "--origins", self.output_origins,
            "--destinations", self.output_destinations,
            "--network-data-source", self.network_data_source,
            "--travel-mode", self.travel_mode,
            "--time-units", self.time_units,
            "--distance-units", self.distance_units,
            "--max-origins", str(self.max_origins),
            "--max-destinations", str(self.max_destinations),
            "--max-processes", str(self.max_processes),
            "--output-format", str(self.output_format_str)
        ]
        # Include correct output location
        if self.output_format is helpers.OutputFormat.featureclass:
            odcm_inputs += ["--output-od-location", self.output_od_lines]
        else:
            odcm_inputs += ["--output-od-location", self.output_data_folder]
        # Include other optional parameters if relevant
        if self.barriers:
            odcm_inputs += ["--barriers"]
            odcm_inputs += self.barriers
        if self.cutoff:
            odcm_inputs += ["--cutoff", str(self.cutoff)]
        if self.num_destinations:
            odcm_inputs += ["--num-destinations", str(self.num_destinations)]
        if self.time_of_day:
            odcm_inputs += ["--time-of-day", self.time_of_day]

        # Run the subprocess
        helpers.execute_subprocess("parallel_odcm.py", odcm_inputs)

    def solve_large_od_cost_matrix(self):
        """Solve the large OD Cost Matrix in parallel."""
        # Set the progressor so the user is informed of progress
        arcpy.SetProgressor("default")

        try:
            arcpy.SetProgressorLabel("Validating inputs...")
            self._validate_inputs()
            arcpy.AddMessage("Inputs successfully validated.")
        except Exception:  # pylint: disable=broad-except
            arcpy.AddError("Invalid inputs.")
            return

        # Preprocess inputs
        arcpy.SetProgressorLabel("Preprocessing inputs...")
        self._preprocess_inputs()

        # Solve the analysis
        arcpy.SetProgressorLabel("Solving analysis in parallel...")
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

    # --output-format parameter
    help_string = ("The desired format for the output OD Cost Matrix Lines results. "
                   f"Choices: {', '.join(helpers.OUTPUT_FORMATS)}")
    parser.add_argument(
        "-of", "--output-format", action="store", dest="output_format", help=help_string, required=True)

    # --output-od-lines parameter
    help_string = ("The catalog path to the output feature class that will contain the combined OD Cost Matrix "
                   "results. Applies only when output-format is 'Feature class'.")
    parser.add_argument(
        "-ol", "--output-od-lines", action="store", dest="output_od_lines", help=help_string, required=False)

    # --output-data-format parameter
    help_string = ("The catalog path to the folder that will contain the OD Cost Matrix result files. "
                   "Applies only when output-format is 'CSV files' or 'Apache Arrow files'.")
    parser.add_argument(
        "-odf", "--output-data-folder", action="store", dest="output_data_folder", help=help_string, required=False)

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

    # --time-of-day parameter
    help_string = (f"The time of day for the analysis. Must be in {helpers.DATETIME_FORMAT} format. Set to None for "
                   "time neutral.")
    parser.add_argument("-tod", "--time-of-day", action="store", dest="time_of_day", help=help_string, required=False)

    # --precalculate-network-locations parameter
    help_string = "Whether or not to precalculate network location fields before solving the analysis."
    parser.add_argument(
        "-pnl", "--precalculate-network-locations", action="store", type=lambda x: bool(strtobool(x)),
        dest="precalculate_network_locations", help=help_string, required=True)

    # --sort-inputs parameter
    help_string = "Whether or not to spatially sort origins and destinations."
    parser.add_argument(
        "-si", "--sort-inputs", action="store", type=lambda x: bool(strtobool(x)),
        dest="sort_inputs", help=help_string, required=True)

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
