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
from concurrent import futures
import os
import sys
import uuid
import logging
import shutil
import itertools
import time
import traceback
import argparse
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










def get_oid_ranges_for_input(input_fc, max_chunk_size):
    """Construct ranges of ObjectIDs for use in where clauses to split large data into chunks.

    Args:
        input_fc (str, layer): Data that needs to be split into chunks
        max_chunk_size (int): Maximum number of rows that can be in a chunk

    Returns:
        list: list of ObjectID ranges for the current dataset representing each chunk. For example,
            [[1, 1000], [1001, 2000], [2001, 2478]] represents three chunks of no more than 1000 rows.
    """
    ranges = []
    num_in_range = 0
    current_range = [0, 0]
    # Loop through all OIDs of the input and construct tuples of min and max OID for each chunk
    # We do it this way and not by straight-up looking at the numerical values of OIDs to account
    # for definition queries, selection sets, or feature layers with gaps in OIDs
    for row in arcpy.da.SearchCursor(input_fc, "OID@"):  # pylint: disable=no-member
        oid = row[0]
        if num_in_range == 0:
            # Starting new range
            current_range[0] = oid
        # Increase the count of items in this range and set the top end of the range to the current oid
        num_in_range += 1
        current_range[1] = oid
        if num_in_range == max_chunk_size:
            # Finishing up a chunk
            ranges.append(current_range)
            # Reset range trackers
            num_in_range = 0
            current_range = [0, 0]
    # After looping, close out the last range if we still have one open
    if current_range != [0, 0]:
        ranges.append(current_range)

    return ranges


class ODCostMatrix:  # pylint:disable = too-many-instance-attributes
    """Used for solving an OD Cost Matrix problem in parallel for a designated chunk of the input datasets."""

    def __init__(self, **kwargs):
        """Initialize the OD Cost Matrix analysis for the given inputs.

        Expected arguments:
        - origins
        - destinations
        - network_data_source
        - travel_mode
        - time_units
        - distance_units
        - cutoff
        - num_destinations
        - output_folder
        - barriers
        """
        self.origins = kwargs["origins"]
        self.destinations = kwargs["destinations"]
        self.network_data_source = kwargs["network_data_source"]
        self.travel_mode = kwargs["travel_mode"]
        self.time_units = kwargs["time_units"]
        self.distance_units = kwargs["distance_units"]
        self.cutoff = kwargs["cutoff"]
        self.num_destinations = kwargs["num_destinations"]
        self.output_folder = kwargs["output_folder"]
        self.barriers = []
        if "barriers" in kwargs:
            self.barriers = kwargs["barriers"]

        # Create a job ID and a folder and scratch gdb for this job
        self.job_id = uuid.uuid4().hex
        self.job_folder = os.path.join(self.output_folder, self.job_id)
        os.mkdir(self.job_folder)
        self.od_workspace = os.path.join(self.job_folder, "scratch.gdb")

        # Setup the class logger. Logs for each parallel process are not written to the console but instead to a
        # process-specific log file.
        self.log_file = os.path.join(self.job_folder, 'ODCostMatrix.log')
        cls_logger = logging.getLogger("ODCostMatrix_" + self.job_id)
        self.setup_logger(cls_logger)
        self.logger = cls_logger

        # Set up other instance attributes
        self.is_service = is_nds_service(self.network_data_source)
        self.od_solver = None
        self.time_attribute = ""
        self.distance_attribute = ""
        self.is_travel_mode_time_based = True
        self.is_travel_mode_dist_based = True
        self.optimized_field_name = None
        self.input_origins_layer = "InputOrigins" + self.job_id
        self.input_destinations_layer = "InputDestinations" + self.job_id
        self.input_origins_layer_obj = None
        self.input_destinations_layer_obj = None

        # Create a network dataset layer
        self.nds_layer_name = "NetworkDatasetLayer"
        if not self.is_service:
            self._make_nds_layer()
            self.network_data_source = self.nds_layer_name

        # Prepare a dictionary to store info about the analysis results
        self.job_result = {
            "jobId": self.job_id,
            "jobFolder": self.job_folder,
            "solveSucceeded": False,
            "solveMessages": "",
            "outputLines": "",
            "logFile": self.log_file
        }

        # Get the ObjectID fields for origins and destinations
        desc_origins = arcpy.Describe(self.origins)
        desc_destinations = arcpy.Describe(self.destinations)
        self.origins_oid_field_name = desc_origins.oidFieldName
        self.destinations_oid_field_name = desc_destinations.oidFieldName
        self.origins_fields = desc_origins.fields
        self.destinations_fields = desc_destinations.fields

    def _make_nds_layer(self):
        """Create a network dataset layer if one does not already exist."""
        if self.is_service:
            return
        if arcpy.Exists(self.nds_layer_name):
            self.logger.debug(f"Using existing network dataset layer: {self.nds_layer_name}")
        else:
            self.logger.debug("Creating network dataset layer...")
            run_gp_tool(
                arcpy.na.MakeNetworkDatasetLayer,
                [self.network_data_source, self.nds_layer_name],
                log_to_use=self.logger
            )

    def initialize_od_solver(self):
        """Initialize an OD solver object and set properties."""
        # For a local network dataset, we need to checkout the Network Analyst extension license.
        if not self.is_service:
            arcpy.CheckOutExtension("network")

        # Create a new OD cost matrix object
        self.logger.debug("Creating OD Cost Matrix object...")
        self.od_solver = arcpy.nax.OriginDestinationCostMatrix(self.network_data_source)

        # Set the OD cost matrix analysis properties.
        # Read properties from the od_config.py config file for all properties not set in the UI as parameters.
        # OD properties documentation: https://pro.arcgis.com/en/pro-app/arcpy/network-analyst/odcostmatrix.htm
        # The properties have been extracted to the config file to make them easier to find and set so users don't have
        # to dig through the code to change them.
        self.logger.debug("Setting OD Cost Matrix analysis properties from OD config file...")
        for prop in OD_PROPS:
            if prop in OD_PROPS_SET_BY_TOOL:
                self.logger.warning(
                    f"OD config file property {prop} is handled explicitly by the tool parameters and will be ignored."
                )
                continue
            try:
                setattr(self.od_solver, prop, OD_PROPS[prop])
            except Exception as ex:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to set property {prop} from OD config file. Default will be used instead.")
                self.logger.warning(str(ex))
        # Set properties explicitly specified in the tool UI as arguments
        self.logger.debug("Setting OD Cost Matrix analysis properties specified tool inputs...")
        self.od_solver.travelMode = self.travel_mode
        self.od_solver.timeUnits = self.time_units
        self.od_solver.distanceUnits = self.distance_units
        self.od_solver.defaultDestinationCount = self.num_destinations
        self.od_solver.defaultImpedanceCutoff = self.cutoff

        # Determine if the travel mode has impedance units that are time-based, distance-based, or other.
        self._determine_if_travel_mode_time_based()

    def solve(self, origins_criteria, destinations_criteria):
        """Create and solve an OD Cost Matrix analysis for the designated chunk of origins and destinations.

        Args:
            origins_criteria (list): Origin ObjectID range to select from the input dataset
            destinations_criteria ([type]): Destination ObjectID range to select from the input dataset
        """
        # Make output gdb
        self.logger.debug("Creating output geodatabase for OD cost matrix analysis...")
        run_gp_tool(
            arcpy.management.CreateFileGDB,
            [os.path.dirname(self.od_workspace), os.path.basename(self.od_workspace)],
            log_to_use=self.logger
        )

        # Select the origins and destinations to process
        self._select_inputs(origins_criteria, destinations_criteria)
        if not self.input_destinations_layer_obj:
            # No destinations met the criteria for this set of origins
            self.logger.debug("No destinations met the criteria for this set of origins. Skipping OD calculation.")
            return

        # Initialize the OD solver object
        self.initialize_od_solver()

        # Load the origins
        self.logger.debug("Loading origins...")
        origins_field_mappings = self.od_solver.fieldMappings(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
            True  # Use network location fields
        )
        self.od_solver.load(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
            self.input_origins_layer_obj,
            origins_field_mappings,
            False
        )

        # Load the destinations
        self.logger.debug("Loading destinations...")
        destinations_field_mappings = self.od_solver.fieldMappings(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
            True  # Use network location fields
        )
        self.od_solver.load(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
            self.input_destinations_layer_obj,
            destinations_field_mappings,
            False
        )

        # Load barriers
        # Note: This loads ALL barrier features for every analysis, even if they are very far away from any of
        # the inputs in the current chunk. You may want to select only barriers within a reasonable distance of the
        # inputs, particularly if you run into the maximumFeaturesAffectedByLineBarriers,
        # maximumFeaturesAffectedByPointBarriers, and maximumFeaturesAffectedByPolygonBarriers tool limits for portal
        # solves. However, since barriers is likely an unusual case, deal with this only if it becomes a problem.
        for barrier_fc in self.barriers:
            self.logger.debug(f"Loading barriers feature class {barrier_fc}...")
            shape_type = arcpy.Describe(barrier_fc).shapeType
            if shape_type == "Polygon":
                class_type = arcpy.nax.OriginDestinationCostMatrixInputDataType.PolygonBarriers
            elif shape_type == "Polyline":
                class_type = arcpy.nax.OriginDestinationCostMatrixInputDataType.LineBarriers
            elif shape_type == "Point":
                class_type = arcpy.nax.OriginDestinationCostMatrixInputDataType.PointBarriers
            else:
                self.logger.warning(
                    f"Barrier feature class {barrier_fc} has an invalid shape type and will be ignored."
                )
                continue
            barriers_field_mappings = self.od_solver.fieldMappings(class_type, True)
            self.od_solver.load(class_type, barrier_fc, barriers_field_mappings, True)

        # Solve the OD cost matrix analysis
        self.logger.debug("Solving OD cost matrix...")
        solve_start = time.time()
        solve_result = self.od_solver.solve()
        solve_end = time.time()
        self.logger.debug(f"Solving OD cost matrix completed in {round(solve_end - solve_start, 3)} (seconds).")

        # Handle solve messages
        solve_msgs = [msg[-1] for msg in solve_result.solverMessages(arcpy.nax.MessageSeverity.All)]
        initial_num_msgs = len(solve_msgs)
        for msg in solve_msgs:
            self.logger.debug(msg)
        # Remove repetitive messages so they don't clog up the stdout pipeline when running the tool
        # 'No "Destinations" found for "Location 1" in "Origins".' is a common message that tends to be repeated and is
        # not particularly useful to see in bulk.
        # Note that this will not work for localized software when this message is translated.
        common_msg_prefix = 'No "Destinations" found for '
        solve_msgs = [msg for msg in solve_msgs if not msg.startswith(common_msg_prefix)]
        num_msgs_removed = initial_num_msgs - len(solve_msgs)
        if num_msgs_removed:
            self.logger.debug(f"Repetitive messages starting with {common_msg_prefix} were consolidated.")
            solve_msgs.append(f"No destinations were found for {num_msgs_removed} origins.")
        solve_msgs = "\n".join(solve_msgs)

        # Update the result dictionary
        self.job_result["solveMessages"] = solve_msgs
        if not solve_result.solveSucceeded:
            self.logger.debug("Solve failed.")
            return
        self.logger.debug("Solve succeeded.")
        self.job_result["solveSucceeded"] = True

        # Export the OD Lines output to a feature class
        output_od_lines = os.path.join(self.od_workspace, "output_od_lines")
        self.logger.debug(f"Exporting OD cost matrix Lines output to {output_od_lines}...")
        solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, output_od_lines)
        self.job_result["outputLines"] = output_od_lines

        self.logger.debug("Finished calculating OD cost matrix.")

    def _hour_to_time_units(self):
        """Convert 1 hour to the user's specified time units.

        Raises:
            ValueError: if the time units are not one of the known arcpy.nax.TimeUnits enums

        Returns:
            float: 1 hour in the user's specified time units
        """
        if self.time_units == arcpy.nax.TimeUnits.Minutes:
            return 60.
        if self.time_units == arcpy.nax.TimeUnits.Seconds:
            return 3600.
        if self.time_units == arcpy.nax.TimeUnits.Hours:
            return 1.
        if self.time_units == arcpy.nax.TimeUnits.Days:
            return 1/24.
        # If we got to this point, the time units were invalid.
        err = f"Invalid time units: {self.time_units}"
        self.logger.error(err)
        raise ValueError(err)

    def _mile_to_dist_units(self):
        """Convert 1 mile to the user's specified distance units.

        Raises:
            ValueError: if the distance units are not one of the known arcpy.nax.DistanceUnits enums

        Returns:
            float: 1 mile in the user's specified distance units
        """
        if self.distance_units == arcpy.nax.DistanceUnits.Miles:
            return 1.
        if self.distance_units == arcpy.nax.DistanceUnits.Kilometers:
            return 1.60934
        if self.distance_units == arcpy.nax.DistanceUnits.Meters:
            return 1609.33999997549
        if self.distance_units == arcpy.nax.DistanceUnits.Feet:
            return 5280.
        if self.distance_units == arcpy.nax.DistanceUnits.Yards:
            return 1760.
        if self.distance_units == arcpy.nax.DistanceUnits.NauticalMiles:
            return 0.868976
        # If we got to this point, the distance units were invalid.
        err = f"Invalid distance units: {self.distance_units}"
        self.logger.error(err)
        raise ValueError(err)

    def _convert_time_cutoff_to_distance(self):
        """Convert a time-based cutoff to distance units

        For a time-based travel mode, the cutoff is expected to be in the user's specified time units. Convert this
        to a safe straight-line distance cutoff in the user's specified distance units to use when pre-selecting
        destinations relevant to this chunk.

        Returns:
            float: Distance cutoff to use for pre-selecting destinations by straight-line distance
        """
        # Assume a max driving speed. Note: If your analysis is doing something other than driving, you may want to
        # update this.
        max_speed = 80.  # Miles per hour
        # Convert the assumed max speed to the user-specified distance units / time units
        max_speed = max_speed * (self._mile_to_dist_units() / self._hour_to_time_units())  # distance units / time units
        # Convert the user's cutoff from time to the user's distance units
        cutoff_dist = self.cutoff * max_speed
        # Add a 5% margin to be on the safe side
        cutoff_dist = cutoff_dist + (0.05 * cutoff_dist)
        return cutoff_dist

    def _select_inputs(self, origins_criteria, destinations_criteria):
        """Create layers from the origins and destinations so the layers contain only the desired inputs for the chunk.

        Args:
            origins_criteria (list): Origin ObjectID range to select from the input dataset
            destinations_criteria ([type]): Destination ObjectID range to select from the input dataset
        """
        # Select the origins with ObjectIDs in this range
        self.logger.debug("Selecting origins for this chunk...")
        origins_where_clause = (
            f"{self.origins_oid_field_name} >= {origins_criteria[0]} "
            f"And {self.origins_oid_field_name} <= {origins_criteria[1]}"
        )
        self.input_origins_layer_obj = run_gp_tool(
            arcpy.management.MakeFeatureLayer,
            [self.origins, self.input_origins_layer, origins_where_clause],
            log_to_use=self.logger
        ).getOutput(0)

        # Select the destinations with ObjectIDs in this range
        self.logger.debug("Selecting destinations for this chunk...")
        destinations_where_clause = (
            f"{self.destinations_oid_field_name} >= {destinations_criteria[0]} "
            f"And {self.destinations_oid_field_name} <= {destinations_criteria[1]} "
        )
        self.input_destinations_layer_obj = run_gp_tool(
            arcpy.management.MakeFeatureLayer,
            [self.destinations, self.input_destinations_layer, destinations_where_clause],
            log_to_use=self.logger
        ).getOutput(0)

        # Eliminate irrelevant destinations in this chunk if possible by selecting only those that fall within a
        # reasonable straight-line distance cutoff. The straight-line distance will always be >= the network distance,
        # so any destinations falling beyond our cutoff limit in straight-line distance are guaranteed to be irrelevant
        # for the network-based OD cost matrix analysis
        # > If not using an impedance cutoff, we cannot do anything here, so just return
        if not self.cutoff:
            return
        # > If using a travel mode with impedance units that are not time or distance-based, we cannot determine how to
        # convert the cutoff units into a sensible distance buffer, so just return
        if not self.is_travel_mode_time_based and not self.is_travel_mode_dist_based:
            return
        # > If using a distance-based travel mode, use the cutoff value directly
        if self.is_travel_mode_dist_based:
            cutoff_dist = self.cutoff + (0.05 * self.cutoff)  # Use 5% margin to be on the safe side
        # > If using a time-based travel mode, convert the time-based cutoff to a distance value in the user's specified
        # distance units by assuming a fast maximum travel speed
        else:
            cutoff_dist = self._convert_time_cutoff_to_distance()

        # Use SelectLayerByLocation to select those within a straight-line distance
        self.logger.debug(
            f"Eliminating destinations outside of distance threshold {cutoff_dist} {self.distance_units.name}...")
        self.input_destinations_layer_obj = run_gp_tool(arcpy.management.SelectLayerByLocation, [
            self.input_destinations_layer,
            "WITHIN_A_DISTANCE_GEODESIC",
            self.input_origins_layer,
            f"{cutoff_dist} {self.distance_units.name}",
        ], log_to_use=self.logger).getOutput(0)

        # If no destinations are within the cutoff, reset the destinations layer object
        # so the iteration will be skipped
        if not self.input_destinations_layer_obj.getSelectionSet():
            self.input_destinations_layer_obj = None
            msg = "No destinations found within the distance threshold."
            self.logger.debug(msg)
            self.job_result["solveMessages"] = msg
            return

    def _determine_if_travel_mode_time_based(self):
        """Determine if the travel mode uses a time-based impedance attribute."""
        # Get the travel mode object from the already-instantiated OD solver object. This saves us from having to parse
        # the user's input travel mode from its string name, object, or json representation.
        travel_mode = self.od_solver.travelMode
        impedance = travel_mode.impedance
        time_attribute = travel_mode.timeAttributeName
        distance_attribute = travel_mode.distanceAttributeName
        self.is_travel_mode_time_based = True if time_attribute == impedance else False
        self.is_travel_mode_dist_based = True if distance_attribute == impedance else False
        # Determine which of the OD Lines output table fields contains the optimized cost values
        if not self.is_travel_mode_time_based and not self.is_travel_mode_dist_based:
            self.optimized_field_name = "Total_Other"
        elif self.is_travel_mode_time_based:
            self.optimized_field_name = "Total_Time"
        else:
            self.optimized_field_name = "Total_Distance"

    def setup_logger(self, logger_obj):
        """Set up the logger used for logging messages for this process. Logs are written to a text file.

        Args:
            logger_obj: The logger instance.
        """
        logger_obj.setLevel(logging.DEBUG)
        if len(logger_obj.handlers) <= 1:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            logger_obj.addHandler(file_handler)
            formatter = logging.Formatter("%(process)d | %(message)s")
            file_handler.setFormatter(formatter)
            logger_obj.addHandler(file_handler)


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


def solve_od_cost_matrix(inputs, chunk):
    """Solve an OD Cost Matrix analysis for the given inputs for the given chunk of ObjectIDs.

    Args:
        inputs (dict): Dictionary of keyword inputs suitable for initializing the ODCostMatrix class
        chunk (list): Represents the ObjectID ranges to select from the origins and destinations when solving the OD
            Cost Matrix. For example, [[1, 1000], [4001, 5000]] means use origin OIDs 1-1000 and destination OIDs
            4001-5000.

    Returns:
        dict: Dictionary of results from the ODCostMatrix class
    """
    odcm = ODCostMatrix(**inputs)
    odcm.logger.info((
        f"Processing origins OID {chunk[0][0]} to {chunk[0][1]} and destinations OID {chunk[1][0]} to {chunk[1][1]} "
        f"as job id {odcm.job_id}"
    ))
    odcm.solve(chunk[0], chunk[1])
    return odcm.job_result


def post_process_od_lines(od_line_fcs, out_fc, num_destinations, sort_field):
    """Merge and post-process the OD Lines calculated in each separate process.

    Args:
        od_line_fcs (list(str)): List of catalog paths to the OD lines outputs from each OD Cost Matrix result. These
            will be combined into one feature class.
        out_fc (str): Catalog path of the output feature class to be created
    """
    LOGGER.info("Post-processing OD Cost Matrix results...")

    # Merge all the individual OD Lines feature classes
    LOGGER.debug("Merging OD Cost Matrix results...")
    run_gp_tool(arcpy.management.Merge, [od_line_fcs, out_fc])

    # If we wanted to find only the k closest destinations for each origin, we have to do additional post-processing.
    # Calculating the OD in chunks means our merged output may have more than k destinations for each origin because
    # each individual chunk found the closest k for that chunk. We need to eliminate all extra rows beyond the first k.
    # Sort the data by OriginOID and the Total_ field that was optimized for the analysis.
    if num_destinations:
        LOGGER.debug("Sorting merged OD Lines results...")
        out_sorted_lines = arcpy.CreateUniqueName("ODLines_Sorted", arcpy.env.scratchGDB)  # pylint: disable=no-member
        sort_fields = [["OriginOID", "ASCENDING"], [sort_field, "ASCENDING"]]
        run_gp_tool(arcpy.management.Sort, [out_fc, out_sorted_lines, sort_fields])
        desc = arcpy.Describe(out_sorted_lines)
        # Delete the original output OD lines feature class and re-create it from scratch with the same schema.
        run_gp_tool(arcpy.management.Delete, [[out_fc]])
        run_gp_tool(arcpy.management.CreateFeatureclass, [
            os.path.dirname(out_fc),
            os.path.basename(out_fc),
            "POLYLINE",
            out_sorted_lines,  # template feature class to transfer full schema
            "SAME_AS_TEMPLATE",
            "SAME_AS_TEMPLATE",
            desc.spatialReference
        ])
        # Loop through the sorted feature class and insert only the first k into the final output
        field_names = ["OriginOID", "SHAPE@"] + [f.name for f in desc.fields if f.name != "OriginOID"]
        with arcpy.da.InsertCursor(out_fc, field_names) as cur:  # pylint: disable=no-member
            current_origin_id = None
            count = 0
            for row in arcpy.da.SearchCursor(out_sorted_lines, field_names):  # pylint: disable=no-member
                origin_id = row[0]
                if origin_id != current_origin_id:
                    # Starting a fresh origin ID
                    current_origin_id = origin_id
                    count = 0
                count += 1
                if count > num_destinations:
                    # Skip this row because we have exceeded the number we want to keep for this origin
                    continue
                # If we got this far, we want to keep this row.
                cur.insertRow(row)

        # Clean up intermediate outputs
        LOGGER.debug("Deleting intermediate post-processing outputs...")
        run_gp_tool(arcpy.management.Delete, [[out_sorted_lines]])

    LOGGER.info("Post-processing complete.")
    LOGGER.info(f"Results written to {out_fc}.")


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
        self.barriers = barriers

        self.same_origins_destinations = True if self.origins == self.destinations else False

        self.max_origins = self.chunk_size
        self.max_destinations = self.chunk_size

        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.service_limits = None
        self.is_agol = False

        # Scratch folder to store intermediate outputs from the OD Cost Matrix processes
        unique_id = uuid.uuid4().hex
        self.scratch_folder = os.path.join(arcpy.env.scratchFolder, "ODCM_" + unique_id)  # pylint: disable=no-member

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
            try:
                arcpy.CheckOutExtension("network")
            except Exception as ex:
                err = "Unable to check out Network Analyst extension license."
                arcpy.AddError(err)
                raise RuntimeError(err) from ex

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

        arcpy.AddMessage(f"Intermediate outputs will be written to {self.scratch_folder}.")
        os.mkdir(self.scratch_folder)

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
            "--network-data-source", helpers.get_catalog_path(self.network_data_source),
            "--travel-mode", get_travel_mode_json(parameters[6]),
            "--time-units", self.time_units,
            "--distance-units", self.distance_units,
            "--max-origins", str(self.max_origins),
            "--max-destinations", str(self.max_destinations),
            "--max-processes", str(self.max_processes),
            "--barriers"
        ] + get_catalog_path_multivalue(parameters[13])
        cutoff = parameters[11].valueAsText
        if cutoff:
            odcm_inputs += ["--cutoff", cutoff]
        num_destinations = parameters[12].valueAsText
        if num_destinations:
            odcm_inputs += ["--num-destinations", num_destinations]
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
                    parse_std_and_write_to_gp_ui(msg_string)
                time.sleep(.5)

            # Once the process is finished, check if any additional errors were returned. Messages that came after the
            # last process.poll() above will still be in the queue here. This is especially important for detecting
            # messages from raised exceptions, especially those with tracebacks.
            output, _ = process.communicate()
            if output:
                out_msgs = output.decode().splitlines()
                for msg in out_msgs:
                    parse_std_and_write_to_gp_ui(msg)

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
