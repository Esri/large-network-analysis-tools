"""Compute a large Origin Destination (OD) cost matrices by chunking the
inputs, solving in parallel, and recombining the results into a single
feature class.

This is a sample script users can modify to fit their specific needs.

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
# pylint: disable=logging-fstring-interpolation, too-many-lines
import os
import sys
import uuid
import logging
import shutil
import itertools
import time
import traceback
import argparse
import subprocess
from distutils.util import strtobool

import arcpy

# Import OD Cost Matrix settings from config file
from od_config import OD_PROPS, OD_PROPS_SET_BY_TOOL

import helpers

arcpy.env.overwriteOutput = True


# Set logging for the main process.
# LOGGER logs everything from the main process to stdout using a specific format that the SolveLargeODCostMatrix tool
# can parse and write to the geoprocessing message feed.
LOG_LEVEL = logging.INFO  # Set to logging.DEBUG to see verbose debug messages
LOGGER = logging.getLogger(__name__)  # pylint:disable=invalid-name
LOGGER.setLevel(LOG_LEVEL)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(LOG_LEVEL)
# Used by script tool to split message text from message level to add correct message type to GP window
MSG_STR_SPLITTER = " | "
console_handler.setFormatter(logging.Formatter("%(levelname)s" + MSG_STR_SPLITTER + "%(message)s"))
LOGGER.addHandler(console_handler)

# Set some global variables. Some of these are also referenced in the script tool definition.
DISTANCE_UNITS = ["Kilometers", "Meters", "Miles", "Yards", "Feet", "NauticalMiles"]
TIME_UNITS = ["Days", "Hours", "Minutes", "Seconds"]
MAX_AGOL_PROCESSES = 4  # AGOL concurrent processes are limited so as not to overload the service for other users.
DELETE_INTERMEDIATE_OD_OUTPUTS = True  # Set to False for debugging purposes


def run_gp_tool(tool, tool_args=None, tool_kwargs=None, log_to_use=LOGGER):
    """Run a geoprocessing tool with nice logging.

    The purpose of this function is simply to wrap the call to a geoprocessing tool in a way that we can log errors,
    warnings, and info messages as well as tool run time into our logging. This helps pipe the messages back to our
    script tool dialog.

    Args:
        tool (arcpy geoprocessing tool class): GP tool class command, like arcpy.management.CreateFileGDB
        tool_args (list, optional): Ordered list of values to use as tool arguments. Defaults to None.
        tool_kwargs (dictionary, optional): Dictionary of tool parameter names and values that can be used as named
            arguments in the tool command. Defaults to None.
        log_to_use (logging.logger, optional): logger class to use for messages. Defaults to LOGGER. When calling this
            from the ODCostMatrix class, use self.logger instead so the messages go to the processes's log file instead
            of stdout.

    Returns:
        GP result object: GP result object returned from the tool run.

    Raises:
        arcpy.ExecuteError if the tool fails
    """
    # Try to retrieve and log the name of the tool
    tool_name = repr(tool)
    try:
        tool_name = tool.__esri_toolname__
    except Exception:  # pylint: disable=broad-except
        try:
            tool_name = tool.__name__
        except Exception:  # pylint: disable=broad-except
            # Probably the tool didn't have an __esri_toolname__ property or __name__. Just don't worry about it.
            pass
    log_to_use.debug(f"Running geoprocessing tool {tool_name}...")

    # Try running the tool, and log all messages
    try:
        if tool_args is None:
            tool_args = []
        if tool_kwargs is None:
            tool_kwargs = {}
        result = tool(*tool_args, **tool_kwargs)
        info_msgs = [msg for msg in result.getMessages(0).splitlines() if msg]
        warning_msgs = [msg for msg in result.getMessages(1).splitlines() if msg]
        for msg in info_msgs:
            log_to_use.debug(msg)
        for msg in warning_msgs:
            log_to_use.warning(msg)
    except arcpy.ExecuteError:
        log_to_use.error(f"Error running geoprocessing tool {tool_name}.")
        # First check if it's a tool error and if so, handle warning and error messages.
        info_msgs = [msg for msg in arcpy.GetMessages(0).strip("\n").splitlines() if msg]
        warning_msgs = [msg for msg in arcpy.GetMessages(1).strip("\n").splitlines() if msg]
        error_msgs = [msg for msg in arcpy.GetMessages(2).strip("\n").splitlines() if msg]
        for msg in info_msgs:
            log_to_use.debug(msg)
        for msg in warning_msgs:
            log_to_use.warning(msg)
        for msg in error_msgs:
            log_to_use.error(msg)
        raise
    except Exception:
        # Unknown non-tool error
        log_to_use.error(f"Error running geoprocessing tool {tool_name}.")
        errs = traceback.format_exc().splitlines()
        for err in errs:
            log_to_use.error(err)
        raise

    log_to_use.debug(f"Finished running geoprocessing tool {tool_name}.")
    return result







def validate_od_settings(**od_inputs):
    """Validate OD cost matrix settings before spinning up a bunch of parallel processes doomed to failure.

    Also check which field name in the output OD Lines will store the optimized cost values. This depends on the travel
    mode being used by the analysis, and we capture it here to use in later steps.

    Returns:
        str: The name of the field in the output OD Lines table containing the optimized costs for the analysis
    """
    # Create a dummy ODCostMatrix object, initialize an OD solver object, and set properties
    # This allows us to detect any errors prior to spinning up a bunch of parallel processes and having them all fail.
    LOGGER.debug("Validating OD Cost Matrix settings...")
    odcm = None
    optimized_cost_field = None
    try:
        odcm = ODCostMatrix(**od_inputs)
        odcm.initialize_od_solver()
        # Check which field name in the output OD Lines will store the optimized cost values
        optimized_cost_field = odcm.optimized_field_name
        LOGGER.debug("OD Cost Matrix settings successfully validated.")
    except Exception:
        LOGGER.error("Error initializing OD Cost Matrix analysis.")
        errs = traceback.format_exc().splitlines()
        for err in errs:
            LOGGER.error(err)
        raise
    finally:
        if odcm:
            LOGGER.debug("Deleting temporary test OD Cost Matrix job folder...")
            shutil.rmtree(odcm.job_result["jobFolder"], ignore_errors=True)

    return optimized_cost_field



class od_cost_matrix_solver():
    """Compute OD Cost Matrices between Origins and Destinations in parallel and combine results.

    Preprocess and validate inputs, compute OD cost matrices in parallel, and combine and post-process the results.
    This method does all the work.

    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, origins, destinations, network_data_source, travel_mode, output_od_lines, output_origins,
        output_destinations, chunk_size, max_processes, time_units, distance_units, cutoff=None, num_destinations=None,
        should_precalc_network_locations=True, barriers=None
    ):
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
        self.should_precalc_network_locations = should_precalc_network_locations
        self.barriers = barriers if barriers else []

        self.same_origins_destinations = True if self.origins == self.destinations else False

        self.max_origins = self.chunk_size
        self.max_destinations = self.chunk_size

        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.service_limits = None
        self.is_agol = False

    def validate_inputs(self):
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
        if self.cutoff and self.cutoff <= 0:
            err = "Impedance cutoff must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.num_destinations and self.num_destinations < 1:
            err = "Number of destinations to find must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)

        # Validate time and distance units
        helpers.convert_time_units_str_to_enum(self.time_units)
        helpers.convert_distance_units_str_to_enum(self.distance_units)

        # Validate origins and destinations
        if not arcpy.Exists(self.origins):
            err = f"Input Origins dataset {self.origins} does not exist."
            arcpy.AddError(err)
            raise ValueError(err)
        if int(arcpy.management.GetCount(self.origins).getOutput(0)) <= 0:
            err = f"Input Origins dataset {self.origins} has no rows."
            arcpy.AddError(err)
            raise ValueError(err)
        if not arcpy.Exists(self.destinations):
            err = f"Input Destinations dataset {self.destinations} does not exist."
            arcpy.AddError(err)
            raise ValueError(err)
        if int(arcpy.management.GetCount(self.destinations).getOutput(0)) <= 0:
            err = f"Input Destinations dataset {self.destinations} has no rows."
            arcpy.AddError(err)
            raise ValueError(err)

        # Validate barriers
        for barrier_fc in self.barriers:
            if not arcpy.Exists(barrier_fc):
                err = f"Input Barriers dataset {barrier_fc} does not exist."
                arcpy.AddError(err)
                raise ValueError(err)

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

        # Validate travel mode and convert it to a string
        self.travel_mode = helpers.get_travel_mode_string(self.travel_mode)

        # For a services solve, get tool limits and validate max processes and chunk size
        if self.is_service:
            self._get_tool_limits_and_is_agol()
            if self.is_agol and self.max_processes > MAX_AGOL_PROCESSES:
                arcpy.AddWarning((
                    f"The specified maximum number of parallel processes, {self.max_processes}, exceeds the limit of "
                    f"{MAX_AGOL_PROCESSES} allowed when using as the network data source the ArcGIS Online services or "
                    "a hybrid portal whose network analysis services fall back to the ArcGIS Online services. The "
                    f"maximum number of parallel processes has been reduced to {MAX_AGOL_PROCESSES}."))
                self.max_processes = MAX_AGOL_PROCESSES
            self._update_max_inputs_for_service()
            if self.should_precalc_network_locations:
                arcpy.AddWarning(
                    "Cannot precalculate network location fields when the network data source is a service.")
                self.should_precalc_network_locations = False

        ### TODO: Figure out how to validate OD settings and get optimized cost field

    def _get_tool_limits_and_is_agol(
            self, service_name="asyncODCostMatrix", tool_name="GenerateOriginDestinationCostMatrix"):
        """Retrieve a dictionary of various limits supported by a portal tool and whether the portal uses AGOL services.

        Assumes that we have already determined that the network data source is a service.

        Args:
            service_name (str, optional): Name of the service. Defaults to "asyncODCostMatrix".
            tool_name (str, optional): Tool name for the designated service. Defaults to
                "GenerateOriginDestinationCostMatrix".
        """
        LOGGER.debug("Getting tool limits from the portal...")
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

    def _spatially_sort_input(self, input_features, is_origins):
        """Spatially sort the input feature class.

        Also adds a field to the input feature class to preserve the original OID values. This field is called
        "OriginOID" for origins and "DestinationOID" for destinations.

        Args:
            input_features (str): Catalog path to the feature class to sort
            is_origins (bool): True if the feature class represents origins; False otherwise.
        """
        LOGGER.info(f"Spatially sorting input dataset {input_features}...")

        # Add a unique ID field so we don't lose OID info when we sort and can use these later in joins.
        # Note that if the original input was a shapefile, these IDs will likely be wrong because copying the original
        # input to the output geodatabase will have altered the original ObjectIDs.
        # Consequently, don't use shapefiles as inputs.
        LOGGER.debug("Transferring original OID values to new field...")
        oid_field = "OriginOID" if is_origins else "DestinationOID"
        desc = arcpy.Describe(input_features)
        if oid_field in [f.name for f in desc.fields]:
            run_gp_tool(arcpy.management.DeleteField, [input_features, oid_field])
        run_gp_tool(arcpy.management.AddField, [input_features, oid_field, "LONG"])
        run_gp_tool(arcpy.management.CalculateField, [input_features, oid_field, f"!{desc.oidFieldName}!"])

        # Make a temporary copy of the inputs so the Sort tool can write its output to the input_features path, which is
        # the ultimate desired location
        temp_inputs = arcpy.CreateUniqueName("TempODInputs", arcpy.env.scratchGDB)  # pylint:disable = no-member
        LOGGER.debug(f"Making temporary copy of inputs in {temp_inputs} before sorting...")
        run_gp_tool(arcpy.management.Copy, [input_features, temp_inputs])

        # Spatially sort input features
        try:
            LOGGER.debug("Running spatial sort...")
            # Don't use run_gp_tool() because we need to parse license errors.
            arcpy.management.Sort(temp_inputs, input_features, [[desc.shapeFieldName, "ASCENDING"]], "PEANO")
        except arcpy.ExecuteError:  # pylint:disable = no-member
            msgs = arcpy.GetMessages(2)
            if "000824" in msgs:  # ERROR 000824: The tool is not licensed.
                LOGGER.warning("Skipping spatial sorting because the Advanced license is not available.")
            else:
                LOGGER.warning(f"Skipping spatial sorting because the tool failed. Messages:\n{msgs}")

        # Clean up. Delete temporary copy of inputs
        LOGGER.debug(f"Deleting temporary input feature class {temp_inputs}...")
        run_gp_tool(arcpy.management.Delete, [[temp_inputs]])

    def _precalculate_network_locations(self, input_features):
        """Precalculate network location fields if possible for faster loading and solving later.

        Cannot be used if the network data source is a service. Uses the searchTolerance, searchToleranceUnits, and
        searchQuery properties set in the OD config file.

        Args:
            input_features (feature class catalog path): Feature class to calculate network locations for
            network_data_source (network dataset catalog path): Network dataset to use to calculate locations
            travel_mode (travel mode): Travel mode name, object, or json representation to use when calculating locations.
        """
        if self.is_service:
            arcpy.AddMessage(
                "Skipping precalculating network location fields because the network data source is a service.")
            return

        LOGGER.info(f"Precalculating network location fields for {input_features}...")

        # Get location settings from config file if present
        search_tolerance = None
        if "searchTolerance" in OD_PROPS and "searchToleranceUnits" in OD_PROPS:
            search_tolerance = f"{OD_PROPS['searchTolerance']} {OD_PROPS['searchToleranceUnits'].name}"
        search_query = None
        if "searchQuery" in OD_PROPS:
            search_query = OD_PROPS["searchQuery"]

        # Calculate network location fields if network data source is local
        run_gp_tool(
            arcpy.na.CalculateLocations,
            [input_features, self.network_data_source],
            {"search_tolerance": search_tolerance, "search_query": search_query, "travel_mode": self.travel_mode}
        )

    def execute_solve(self):

        #### TODO: Figure out how to do logging
        # Copy Origins and Destinations to outputs
        LOGGER.debug("Copying input origins and destinations to outputs...")
        run_gp_tool(arcpy.management.Copy, [self.origins, self.output_origins])
        if not self.same_origins_destinations:
            run_gp_tool(arcpy.management.Copy, [self.destinations, self.output_destinations])

        # Spatially sort inputs
        self._spatially_sort_input(self.output_origins, is_origins=True)
        if not self.same_origins_destinations:
            self._spatially_sort_input(self.output_destinations, is_origins=False)

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
            run_gp_tool(arcpy.management.Copy, [self.output_origins, self.output_destinations])

        # Launch the odcm script as a subprocess so it can spawn parallel processes. We have to do this because a tool
        # running in the Pro UI cannot call concurrent.futures without opening multiple instances of Pro.
        cwd = os.path.dirname(os.path.abspath(__file__))
        odcm_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(cwd, "odcm.py"),
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
            "--barriers"
        ] + self.barriers
        if self.cutoff:
            odcm_inputs += ["--cutoff", self.cutoff]
        if self.num_destinations:
            odcm_inputs += ["--num-destinations", self.num_destinations]
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


def solve_large_od_cost_matrix(  # pylint: disable=too-many-locals, too-many-arguments
        origins, destinations, network_data_source, travel_mode, output_od_lines, output_origins,
        output_destinations, chunk_size, max_processes, time_units, distance_units, cutoff=None, num_destinations=None,
        should_precalc_network_locations=True, barriers=None
):
    # Instantiate an od_cost_matrix_solver class
    od_solver = od_cost_matrix_solver(
        origins, destinations, network_data_source, travel_mode, output_od_lines, output_origins,
        output_destinations, chunk_size, max_processes, time_units, distance_units, cutoff, num_destinations,
        should_precalc_network_locations, barriers
    )

    try:
        od_solver.validate_inputs()
        arcpy.AddMessage("Inputs successfully validated.")
    except Exception:
        ## TODO: Double check this
        arcpy.AddError("Invalid inputs.")
        raise arcpy.ExecuteError("Invalid inputs.")


def _launch_tool():
    """Read arguments from the command line (or passed in via subprocess) and run the tool."""
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
        "A JSON string representation of a travel mode from the network data source that will be used for the analysis."
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

    # Call the main execution
    start_time = time.time()
    compute_ods_in_parallel(**args)
    LOGGER.info(f"Completed in {round((time.time() - start_time) / 60, 2)} minutes")


if __name__ == "__main__":
    # The script tool calls this script as if it were calling it from the command line.
    # It uses this main function.
    _launch_tool()
