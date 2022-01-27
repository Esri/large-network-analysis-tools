"""Compute a large Origin Destination (OD) cost matrices by chunking the
inputs, solving in parallel, and recombining the results into a single
feature class.

This is a sample script users can modify to fit their specific needs.

This script is intended to be called as a subprocess from the solve_large_odcm.py script
so that it can launch parallel processes with concurrent.futures. It must be
called as a subprocess because the main script tool process, when running
within ArcGIS Pro, cannot launch parallel subprocesses on its own.

This script should not be called directly from the command line.

Copyright 2022 Esri
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
# pylint: disable=logging-fstring-interpolation
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
console_handler.setFormatter(logging.Formatter("%(levelname)s" + helpers.MSG_STR_SPLITTER + "%(message)s"))
LOGGER.addHandler(console_handler)

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


class ODCostMatrix:  # pylint:disable = too-many-instance-attributes
    """Used for solving an OD Cost Matrix problem in parallel for a designated chunk of the input datasets."""

    def __init__(self, **kwargs):
        """Initialize the OD Cost Matrix analysis for the given inputs.

        Expected arguments:
        - origins
        - destinations
        - output_format
        - output_od_location
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
        self.output_format = kwargs["output_format"]
        self.output_od_location = kwargs["output_od_location"]
        self.network_data_source = kwargs["network_data_source"]
        self.travel_mode = kwargs["travel_mode"]
        self.time_units = kwargs["time_units"]
        self.distance_units = kwargs["distance_units"]
        self.cutoff = kwargs["cutoff"]
        self.num_destinations = kwargs["num_destinations"]
        self.scratch_folder = kwargs["scratch_folder"]
        self.barriers = []
        if "barriers" in kwargs:
            self.barriers = kwargs["barriers"]

        # Create a job ID and a folder and scratch gdb for this job
        self.job_id = uuid.uuid4().hex
        self.job_folder = os.path.join(self.scratch_folder, self.job_id)
        os.mkdir(self.job_folder)

        # Setup the class logger. Logs for each parallel process are not written to the console but instead to a
        # process-specific log file.
        self.log_file = os.path.join(self.job_folder, 'ODCostMatrix.log')
        cls_logger = logging.getLogger("ODCostMatrix_" + self.job_id)
        self.setup_logger(cls_logger)
        self.logger = cls_logger

        # Set up other instance attributes
        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.od_solver = None
        self.solve_result = None
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

    def solve(self, origins_criteria, destinations_criteria):  # pylint: disable=too-many-locals, too-many-statements
        """Create and solve an OD Cost Matrix analysis for the designated chunk of origins and destinations.

        Args:
            origins_criteria (list): Origin ObjectID range to select from the input dataset
            destinations_criteria ([type]): Destination ObjectID range to select from the input dataset
        """
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
        origin_oid_field = "Orig_Origin_OID"
        self.od_solver.addFields(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
            [[origin_oid_field, "LONG"]]
        )
        origins_field_mappings = self.od_solver.fieldMappings(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
            True  # Use network location fields
        )
        for fname in origins_field_mappings:
            if fname == origin_oid_field:
                origins_field_mappings[fname].mappedFieldName = self.origins_oid_field_name
            else:
                origins_field_mappings[fname].mappedFieldName = fname
        self.od_solver.load(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
            self.input_origins_layer_obj,
            origins_field_mappings,
            False
        )

        # Load the destinations
        self.logger.debug("Loading destinations...")
        dest_oid_field = "Orig_Dest_OID"
        self.od_solver.addFields(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
            [[dest_oid_field, "LONG"]]
        )
        destinations_field_mappings = self.od_solver.fieldMappings(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
            True  # Use network location fields
        )
        for fname in destinations_field_mappings:
            if fname == dest_oid_field:
                destinations_field_mappings[fname].mappedFieldName = self.destinations_oid_field_name
            else:
                destinations_field_mappings[fname].mappedFieldName = fname
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
        self.solve_result = self.od_solver.solve()
        solve_end = time.time()
        self.logger.debug(f"Solving OD cost matrix completed in {round(solve_end - solve_start, 3)} (seconds).")

        # Handle solve messages
        solve_msgs = [msg[-1] for msg in self.solve_result.solverMessages(arcpy.nax.MessageSeverity.All)]
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
        if not self.solve_result.solveSucceeded:
            self.logger.debug("Solve failed.")
            return
        self.logger.debug("Solve succeeded.")
        self.job_result["solveSucceeded"] = True

        # Save output
        if self.output_format is helpers.OutputFormat.featureclass:
            self._export_od_lines()
        elif self.output_format is helpers.OutputFormat.csv:
            out_csv_file = os.path.join(
                self.output_od_location,
                (f"ODLines_O_{origins_criteria[0]}_{origins_criteria[1]}_"
                 f"D_{destinations_criteria[0]}_{destinations_criteria[1]}.csv")  # ODLines_O_1_1000_D_2001_3000.csv
            )
            self._export_to_csv(out_csv_file)
        elif self.output_format is helpers.OutputFormat.arrow:
            out_arrow_file = os.path.join(
                self.output_od_location,
                (f"ODLines_O_{origins_criteria[0]}_{origins_criteria[1]}_"
                 f"D_{destinations_criteria[0]}_{destinations_criteria[1]}.arrow")  # ODLines_O_1_1000_D_2001_3000.arrow
            )
            self._export_to_arrow(out_arrow_file)

        self.logger.debug("Finished calculating OD cost matrix.")

    def _export_od_lines(self):
        """Export the OD Lines result to a feature class."""
        # Make output gdb
        od_workspace = os.path.join(self.job_folder, "scratch.gdb")
        self.logger.debug("Creating output geodatabase for OD cost matrix analysis...")
        run_gp_tool(
            arcpy.management.CreateFileGDB,
            [os.path.dirname(od_workspace), os.path.basename(od_workspace)],
            log_to_use=self.logger
        )

        output_od_lines = os.path.join(od_workspace, "output_od_lines")
        self.logger.debug(f"Exporting OD cost matrix Lines output to {output_od_lines}...")
        self.solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, output_od_lines)
        self.job_result["outputLines"] = output_od_lines

        # For services solve, export Origins and Destinations and properly populate OriginOID and DestinationOID fields
        # in the output Lines. Services do not preserve the original input OIDs, instead resetting from 1, unlike solves
        # using a local network dataset, so this extra post-processing step is necessary.
        if self.is_service:
            output_origins = os.path.join(od_workspace, "output_od_origins")
            self.logger.debug(f"Exporting OD cost matrix Origins output to {output_origins}...")
            self.solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Origins, output_origins)
            output_destinations = os.path.join(od_workspace, "output_od_destinations")
            self.logger.debug(f"Exporting OD cost matrix Destinations output to {output_destinations}...")
            self.solve_result.export(
                arcpy.nax.OriginDestinationCostMatrixOutputDataType.Destinations, output_destinations)
            self.logger.debug("Updating values for OriginOID and DestinationOID fields...")
            lines_layer_name = "Out_Lines"
            origins_layer_name = "Out_Origins"
            destinations_layer_name = "Out_Destinations"
            run_gp_tool(arcpy.management.MakeFeatureLayer, [output_od_lines, lines_layer_name], log_to_use=self.logger)
            run_gp_tool(arcpy.management.MakeFeatureLayer, [output_origins, origins_layer_name], log_to_use=self.logger)
            run_gp_tool(arcpy.management.MakeFeatureLayer,
                        [output_destinations, destinations_layer_name], log_to_use=self.logger)
            # Update OriginOID values
            run_gp_tool(
                arcpy.management.AddJoin,
                [lines_layer_name, "OriginOID", origins_layer_name, "ObjectID", "KEEP_ALL"]
            )
            run_gp_tool(
                arcpy.management.CalculateField,
                [lines_layer_name, f"{os.path.basename(output_od_lines)}.OriginOID",
                 f"!{os.path.basename(output_origins)}.{origin_oid_field}!", "PYTHON3"]
            )
            run_gp_tool(arcpy.management.RemoveJoin, [lines_layer_name])
            # Update DestinationOID values
            run_gp_tool(
                arcpy.management.AddJoin,
                [lines_layer_name, "DestinationOID", destinations_layer_name, "ObjectID", "KEEP_ALL"]
            )
            run_gp_tool(
                arcpy.management.CalculateField,
                [lines_layer_name, f"{os.path.basename(output_od_lines)}.DestinationOID",
                 f"!{os.path.basename(output_destinations)}.{dest_oid_field}!", "PYTHON3"]
            )
            run_gp_tool(arcpy.management.RemoveJoin, [lines_layer_name])

    def _export_to_csv(self, out_csv_file):
        """Save the OD Lines result to a CSV file."""
        self.logger.debug(f"Saving OD cost matrix Lines output to CSV as {out_csv_file}.")
        with open(out_csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["OriginOID", "DestinationOID"])
            for row in self.solve_result.searchCursor(
                arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, ["OriginOID", "DestinationOID"]  # TODO: Fields
            ):
                writer.writerow(row)

        ## TODO: Deal with services

    def _export_to_arrow(self, out_arrow_file):
        """Save the OD Lines result to an Apache Arrow file."""
        self.logger.debug(f"Saving OD cost matrix Lines output to Apache Arrow as {out_arrow_file}.")
        self.solve_result.toArrowTable(
            arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines,
            ["OriginOID", "DestinationOID"],  ## TODO: What fields?
            out_arrow_file
        )

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
        """Convert a time-based cutoff to distance units.

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


class ParallelODCalculator():
    """Solves a large OD Cost Matrix by chunking the problem, solving in parallel, and combining results."""

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, origins, destinations, network_data_source, travel_mode, output_format, output_od_location,
        max_origins, max_destinations, max_processes, time_units, distance_units,
        cutoff=None, num_destinations=None, barriers=None
    ):
        """Compute OD Cost Matrices between Origins and Destinations in parallel and combine results.

        Compute OD cost matrices in parallel and combine and post-process the results.
        This class assumes that the inputs have already been pre-processed and validated.

        Args:
            origins (str): Catalog path to origins
            destinations (str): Catalog path to destinations
            network_data_source (str): Network data source catalog path or URL
            travel_mode (str): String-based representation of a travel mode (name or JSON)
            output_format (str): String representation of the output format
            output_od_location (str): Catalog path to the output feature class or folder where the OD Lines output will
                be stored.
            max_origins (int): Maximum origins allowed in a chunk
            max_destinations (int): Maximum destinations allowed in a chunk
            max_processes (int): Maximum number of parallel processes allowed
            time_units (str): String representation of time units
            distance_units (str): String representation of distance units
            cutoff (float, optional): Impedance cutoff to limit the OD Cost Matrix solve. Interpreted in the time_units
                if the travel mode is time-based. Interpreted in the distance-units if the travel mode is distance-
                based. Interpreted in the impedance units if the travel mode is neither time- nor distance-based.
                Defaults to None. When None, do not use a cutoff.
            num_destinations (int, optional): The number of destinations to find for each origin. Defaults to None,
                which means to find all destinations.
            barriers (list(str), optional): List of catalog paths to point, line, and polygon barriers to use.
                Defaults to None.
        """
        time_units = helpers.convert_time_units_str_to_enum(time_units)
        distance_units = helpers.convert_distance_units_str_to_enum(distance_units)
        self.output_format = helpers.convert_output_format_str_to_enum(output_format)
        self.output_od_location = output_od_location
        if cutoff == "":
            cutoff = None
        if not barriers:
            barriers = []
        if num_destinations == "":
            num_destinations = None
        self.num_destinations = num_destinations
        self.max_processes = max_processes

        # Scratch folder to store intermediate outputs from the OD Cost Matrix processes
        unique_id = uuid.uuid4().hex
        self.scratch_folder = os.path.join(arcpy.env.scratchFolder, "ODCM_" + unique_id)  # pylint: disable=no-member
        LOGGER.info(f"Intermediate outputs will be written to {self.scratch_folder}.")
        os.mkdir(self.scratch_folder)

        # Output folder to store CSV or Arrow outputs
        if self.output_format is not helpers.OutputFormat.featureclass:
            if not os.path.exists(self.output_od_location):
                os.mkdir(self.output_od_location)

        # Initialize the dictionary of inputs to send to each OD solve
        self.od_inputs = {
            "origins": origins,
            "destinations": destinations,
            "output_format": self.output_format,
            "output_od_location": self.output_od_location,
            "network_data_source": network_data_source,
            "travel_mode": travel_mode,
            "scratch_folder": self.scratch_folder,
            "time_units": time_units,
            "distance_units": distance_units,
            "cutoff": cutoff,
            "num_destinations": self.num_destinations,
            "barriers": barriers
        }

        # Final combined output feature class
        self.output_od_lines = output_od_lines
        # List of intermediate output feature classes created by each process
        self.od_line_fcs = []

        # Construct OID ranges for chunks of origins and destinations
        origin_ranges = self._get_oid_ranges_for_input(origins, max_origins)
        destination_ranges = self._get_oid_ranges_for_input(destinations, max_destinations)

        # Construct pairs of chunks to ensure that each chunk of origins is matched with each chunk of destinations
        self.ranges = itertools.product(origin_ranges, destination_ranges)
        # Calculate the total number of jobs to use in logging
        self.total_jobs = len(origin_ranges) * len(destination_ranges)

        self.optimized_cost_field = None

    def _validate_od_settings(self):
        """Validate OD cost matrix settings before spinning up a bunch of parallel processes doomed to failure.

        Also check which field name in the output OD Lines will store the optimized cost values. This depends on the
        travel mode being used by the analysis, and we capture it here to use in later steps.

        Returns:
            str: The name of the field in the output OD Lines table containing the optimized costs for the analysis
        """
        # Create a dummy ODCostMatrix object, initialize an OD solver object, and set properties. This allows us to
        # detect any errors prior to spinning up a bunch of parallel processes and having them all fail.
        LOGGER.debug("Validating OD Cost Matrix settings...")
        optimized_cost_field = None
        odcm = None
        try:
            odcm = ODCostMatrix(**self.od_inputs)
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
                del odcm

        return optimized_cost_field

    @staticmethod
    def _get_oid_ranges_for_input(input_fc, max_chunk_size):
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

    def solve_od_in_parallel(self):
        """Solve the OD Cost Matrix in chunks and post-process the results."""
        # Validate OD Cost Matrix settings. Essentially, create a dummy ODCostMatrix class instance and set up the
        # solver object to ensure this at least works. Do this up front before spinning up a bunch of parallel processes
        # the optimized that are guaranteed to all fail. While we're doing this, check and store the field name that
        # will represent costs in the output OD Lines table. We'll use this in post processing.
        self.optimized_cost_field = self._validate_od_settings()

        # Compute OD cost matrix in parallel
        completed_jobs = 0  # Track the number of jobs completed so far to use in logging
        # Use the concurrent.futures ProcessPoolExecutor to spin up parallel processes that solve the OD cost matrices
        with futures.ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            # Each parallel process calls the solve_od_cost_matrix() function with the od_inputs dictionary for the
            # given origin and destination OID ranges.
            jobs = {executor.submit(solve_od_cost_matrix, self.od_inputs, range): range for range in self.ranges}
            # As each job is completed, add some logging information and store the results to post-process later
            for future in futures.as_completed(jobs):
                completed_jobs += 1
                LOGGER.info(
                    f"Finished OD Cost Matrix calculation {completed_jobs} of {self.total_jobs}.")
                try:
                    # The OD cost matrix job returns a results dictionary. Retrieve it.
                    result = future.result()
                except Exception:
                    # If we couldn't retrieve the result, some terrible error happened. Log it.
                    LOGGER.error("Failed to get OD Cost Matrix result from parallel processing.")
                    errs = traceback.format_exc().splitlines()
                    for err in errs:
                        LOGGER.error(err)
                    raise

                # Parse the results dictionary and store components for post-processing.
                if result["solveSucceeded"]:
                    self.od_line_fcs.append(result["outputLines"])
                else:
                    LOGGER.warning(f"Solve failed for job id {result['jobId']}")
                    msgs = result["solveMessages"]
                    LOGGER.warning(msgs)

        # Merge individual OD Lines feature classes into a single feature class
        if self.od_line_fcs:
            self._post_process_od_lines()
        else:
            LOGGER.warning("All OD Cost Matrix solves failed, so no output was produced.")

        # Cleanup
        # Delete the job folders if the job succeeded
        if DELETE_INTERMEDIATE_OD_OUTPUTS:
            LOGGER.info("Deleting intermediate outputs...")
            try:
                shutil.rmtree(self.scratch_folder, ignore_errors=True)
            except Exception:  # pylint: disable=broad-except
                # If deletion doesn't work, just throw a warning and move on. This does not need to kill the tool.
                LOGGER.warning(f"Unable to delete intermediate OD Cost Matrix output folder {self.scratch_folder}.")

        LOGGER.info("Finished calculating OD Cost Matrices.")

    def _post_process_od_lines(self):
        """Merge and post-process the OD Lines calculated in each separate process.

        Args:
            out_fc (str): Catalog path of the output feature class to be created
        """
        LOGGER.info("Post-processing OD Cost Matrix results...")

        # Merge all the individual OD Lines feature classes
        LOGGER.debug("Merging OD Cost Matrix results...")
        run_gp_tool(arcpy.management.Merge, [self.od_line_fcs, self.output_od_lines])

        # If we wanted to find only the k closest destinations for each origin, we have to do additional post-
        # processing. Calculating the OD in chunks means our merged output may have more than k destinations for each
        # origin because each individual chunk found the closest k for that chunk. We need to eliminate all extra rows
        # beyond the first k. Sort the data by OriginOID and the Total_ field that was optimized for the analysis.
        if self.num_destinations:
            LOGGER.debug("Sorting merged OD Lines results...")
            out_sorted_lines = arcpy.CreateUniqueName(
                "ODLines_Sorted", arcpy.env.scratchGDB)  # pylint: disable=no-member
            sort_fields = [["OriginOID", "ASCENDING"], [self.optimized_cost_field, "ASCENDING"]]
            run_gp_tool(arcpy.management.Sort, [self.output_od_lines, out_sorted_lines, sort_fields])
            desc = arcpy.Describe(out_sorted_lines)
            # Delete the original output OD lines feature class and re-create it from scratch with the same schema.
            run_gp_tool(arcpy.management.Delete, [[self.output_od_lines]])
            run_gp_tool(arcpy.management.CreateFeatureclass, [
                os.path.dirname(self.output_od_lines),
                os.path.basename(self.output_od_lines),
                "POLYLINE",
                out_sorted_lines,  # template feature class to transfer full schema
                "SAME_AS_TEMPLATE",
                "SAME_AS_TEMPLATE",
                desc.spatialReference
            ])
            # Loop through the sorted feature class and insert only the first k into the final output
            field_names = ["OriginOID", "SHAPE@"] + [f.name for f in desc.fields if f.name != "OriginOID"]
            with arcpy.da.InsertCursor(self.output_od_lines, field_names) as cur:  # pylint: disable=no-member
                current_origin_id = None
                count = 0
                for row in arcpy.da.SearchCursor(out_sorted_lines, field_names):  # pylint: disable=no-member
                    origin_id = row[0]
                    if origin_id != current_origin_id:
                        # Starting a fresh origin ID
                        current_origin_id = origin_id
                        count = 0
                    count += 1
                    if count > self.num_destinations:
                        # Skip this row because we have exceeded the number we want to keep for this origin
                        continue
                    # If we got this far, we want to keep this row.
                    cur.insertRow(row)

            # Clean up intermediate outputs
            LOGGER.debug("Deleting intermediate post-processing outputs...")
            run_gp_tool(arcpy.management.Delete, [[out_sorted_lines]])

        LOGGER.info("Post-processing complete.")
        LOGGER.info(f"Results written to {self.output_od_lines}.")


def launch_parallel_od():
    """Read arguments passed in via subprocess and run the parallel OD Cost Matrix.

    This script is intended to be called via subprocess via the solve_large_odcm.py module, which does essential
    preprocessing and validation. Users should not call this script directly from the command line

    We must launch this script via subprocess in order to support parallel processing from an ArcGIS Pro script tool,
    which cannot do parallel processing directly.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description=globals().get("__doc__", ""), fromfile_prefix_chars='@')

    # Define Arguments supported by the command line utility

    # --origins parameter
    help_string = "The full catalog path to the feature class containing the origins."
    parser.add_argument("-o", "--origins", action="store", dest="origins", help=help_string, required=True)

    # --destinations parameter
    help_string = "The full catalog path to the feature class containing the destinations."
    parser.add_argument("-d", "--destinations", action="store", dest="destinations", help=help_string, required=True)

    # --output-format parameter
    help_string = ("The desired format for the output OD Cost Matrix Lines results. "
                   f"Choices: {', '.join(helpers.OUTPUT_FORMATS)}")
    parser.add_argument(
        "-of", "--output-format", action="store", dest="output_format", help=help_string, required=True)

    # ----output-od-location parameter
    help_string = "The catalog path to the output feature class or folder that will contain the OD Cost Matrix results."
    parser.add_argument(
        "-ol", "--output-od-location", action="store", dest="output_od_location", help=help_string, required=True)

    # --network-data-source parameter
    help_string = "The full catalog path to the network dataset or a portal url that will be used for the analysis."
    parser.add_argument(
        "-n", "--network-data-source", action="store", dest="network_data_source", help=help_string, required=True)

    # --travel-mode parameter
    help_string = (
        "The name or JSON string representation of the travel mode from the network data source that will be used for "
        "the analysis."
    )
    parser.add_argument("-tm", "--travel-mode", action="store", dest="travel_mode", help=help_string, required=True)

    # --time-units parameter
    help_string = "String name of the time units for the analysis. These units will be used in the output."
    parser.add_argument("-tu", "--time-units", action="store", dest="time_units", help=help_string, required=True)

    # --distance-units parameter
    help_string = "String name of the distance units for the analysis. These units will be used in the output."
    parser.add_argument(
        "-du", "--distance-units", action="store", dest="distance_units", help=help_string, required=True)

    # --max-origins parameter
    help_string = (
        "Maximum number of origins that can be in one chunk for parallel processing of OD Cost Matrix solves. "
        "For example, 1000 means that a chunk consists of no more than 1000 origins and max-destination destinations."
    )
    parser.add_argument(
        "-mo", "--max-origins", action="store", dest="max_origins", type=int, help=help_string, required=True)

    # --max-destinations parameter
    help_string = (
        "Maximum number of destinations that can be in one chunk for parallel processing of OD Cost Matrix solves. "
        "For example, 1000 means that a chunk consists of no more than max-origin origins and 1000 destinations."
    )
    parser.add_argument(
        "-md", "--max-destinations", action="store", dest="max_destinations", type=int, help=help_string, required=True)

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

    # --barriers parameter
    help_string = "A list of catalog paths to the feature classes containing barriers to use in the OD Cost Matrix."
    parser.add_argument(
        "-b", "--barriers", action="store", dest="barriers", help=help_string, nargs='*', required=False)

    # Get arguments as dictionary.
    args = vars(parser.parse_args())

    # Initialize a parallel OD Cost Matrix calculator class
    od_calculator = ParallelODCalculator(**args)
    # Solve the OD Cost Matrix in parallel chunks
    start_time = time.time()
    od_calculator.solve_od_in_parallel()
    LOGGER.info(f"Parallel OD Cost Matrix calculation completed in {round((time.time() - start_time) / 60, 2)} minutes")


if __name__ == "__main__":
    # This script should always be launched via subprocess as if it were being called from the command line.
    launch_parallel_od()
