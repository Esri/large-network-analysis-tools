"""TODO: Compute a large Origin Destination (OD) cost matrix by chunking the
inputs and solving in parallel. Write outputs into a single combined
feature class, a collection of CSV files, or a collection of Apache
Arrow files.

This is a sample script users can modify to fit their specific needs.

This script is intended to be called as a subprocess from the solve_large_rt.py script
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
import datetime
import traceback
import argparse
import csv
import pandas as pd

import arcpy

# Import Route settings from config file
from rt_config import RT_PROPS, RT_PROPS_SET_BY_TOOL

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

DELETE_INTERMEDIATE_OUTPUTS = True  # Set to False for debugging purposes


def run_gp_tool(tool, tool_args=None, tool_kwargs=None, log_to_use=LOGGER):
    ## TODO: Make shared helper function
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


class Route:  # pylint:disable = too-many-instance-attributes
    """Used for solving an Route problem in parallel for a designated chunk of the input datasets."""

    def __init__(self, **kwargs):
        """Initialize the Route analysis for the given inputs.

        Expected arguments:  TODO
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
        - scratch_folder
        - barriers
        """
        self.origins = kwargs["origins"]
        self.origin_id_field = kwargs["origin_id_field"]
        self.assigned_dest_field = kwargs["assigned_dest_field"]
        self.destinations = kwargs["destinations"]
        self.dest_id_field = kwargs["dest_id_field"]
        self.network_data_source = kwargs["network_data_source"]
        self.travel_mode = kwargs["travel_mode"]
        self.time_units = kwargs["time_units"]
        self.distance_units = kwargs["distance_units"]
        self.time_of_day = kwargs["time_of_day"]
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
        self.log_file = os.path.join(self.job_folder, 'RoutePairs.log')
        cls_logger = logging.getLogger("RoutePairs_" + self.job_id)
        self.setup_logger(cls_logger)
        self.logger = cls_logger

        # Set up other instance attributes
        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.rt_solver = None
        self.solve_result = None
        self.input_origins_layer = "InputOrigins" + self.job_id
        self.input_destinations_layer = "InputDestinations" + self.job_id
        self.input_origins_layer_obj = None

        # Create a network dataset layer if needed
        if not self.is_service:
            self._make_nds_layer()

        # Prepare a dictionary to store info about the analysis results
        self.job_result = {
            "jobId": self.job_id,
            "jobFolder": self.job_folder,
            "solveSucceeded": False,
            "solveMessages": "",
            "outputRoutes": "",
            "logFile": self.log_file
        }

    def _make_nds_layer(self):
        """Create a network dataset layer if one does not already exist."""
        if self.is_service:
            return
        nds_layer_name = os.path.basename(self.network_data_source)
        if arcpy.Exists(nds_layer_name):
            # The network dataset layer already exists in this process, so we can re-use it without having to spend
            # time re-opening the network dataset and making a fresh layer.
            self.logger.debug(f"Using existing network dataset layer: {nds_layer_name}")
        else:
            # The network dataset layer does not exist in this process, so create the layer.
            self.logger.debug("Creating network dataset layer...")
            run_gp_tool(
                arcpy.na.MakeNetworkDatasetLayer,
                [self.network_data_source, nds_layer_name],
                log_to_use=self.logger
            )
        self.network_data_source = nds_layer_name

    def initialize_rt_solver(self):
        """Initialize a Route solver object and set properties."""
        # For a local network dataset, we need to checkout the Network Analyst extension license.
        if not self.is_service:
            arcpy.CheckOutExtension("network")

        # Create a new Route object
        self.logger.debug("Creating Route object...")
        self.rt_solver = arcpy.nax.Route(self.network_data_source)

        # Set the Route analysis properties.
        # Read properties from the rt_config.py config file for all properties not set in the UI as parameters.
        # Route properties documentation: https://pro.arcgis.com/en/pro-app/latest/arcpy/network-analyst/route.htm
        # The properties have been extracted to the config file to make them easier to find and set so users don't have
        # to dig through the code to change them.
        self.logger.debug("Setting Route analysis properties from OD config file...")
        for prop in RT_PROPS:
            if prop in RT_PROPS_SET_BY_TOOL:
                self.logger.warning((
                    f"Route config file property {prop} is handled explicitly by the tool parameters and will be "
                    "ignored."
                ))
                continue
            try:
                setattr(self.rt_solver, prop, RT_PROPS[prop])
                if hasattr(RT_PROPS[prop], "name"):
                    self.logger.debug(f"{prop}: {RT_PROPS[prop].name}")
                else:
                    self.logger.debug(f"{prop}: {RT_PROPS[prop]}")
            except Exception as ex:  # pylint: disable=broad-except
                self.logger.warning(f"Failed to set property {prop} from OD config file. Default will be used instead.")
                self.logger.warning(str(ex))
        # Set properties explicitly specified in the tool UI as arguments
        self.logger.debug("Setting Route analysis properties specified tool inputs...")
        self.rt_solver.travelMode = self.travel_mode
        self.logger.debug(f"travelMode: {self.travel_mode}")
        self.rt_solver.timeUnits = self.time_units
        self.logger.debug(f"timeUnits: {self.time_units}")
        self.rt_solver.distanceUnits = self.distance_units
        self.logger.debug(f"distanceUnits: {self.distance_units}")
        self.rt_solver.timeOfDay = self.time_of_day
        self.logger.debug(f"timeOfDay: {self.time_of_day}")

        # Determine if the travel mode has impedance units that are time-based, distance-based, or other.
        self._determine_if_travel_mode_time_based()

    def solve(self, origins_criteria):  # pylint: disable=too-many-locals, too-many-statements
        """Create and solve an Route analysis for the designated chunk of origins and their assigned destinations.

        Args:
            origins_criteria (list): ObjectID range to select from the input origins
        """
        # Select the origins to process
        self._select_inputs(origins_criteria)

        # Initialize the Route solver object
        self.initialize_rt_solver()

        # Insert the origins and destinations
        self._insert_stops()

        if self.rt_solver.count(arcpy.nax.RouteInputDataType.Stops) == 0:
            # There were no valid destinations for this set of origins
            self.logger.debug("No valid destinations for this set of origins. Skipping Route calculation.")
            return

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
            barriers_field_mappings = self.rt_solver.fieldMappings(class_type, True)
            self.rt_solver.load(class_type, barrier_fc, barriers_field_mappings, True)

        # Solve the Route analysis
        self.logger.debug("Solving Route...")
        solve_start = time.time()
        self.solve_result = self.rt_solver.solve()
        solve_end = time.time()
        self.logger.debug(f"Solving Route completed in {round(solve_end - solve_start, 3)} seconds.")

        # Handle solve messages
        solve_msgs = [msg[-1] for msg in self.solve_result.solverMessages(arcpy.nax.MessageSeverity.All)]
        for msg in solve_msgs:
            self.logger.debug(msg)

        # Update the result dictionary
        self.job_result["solveMessages"] = solve_msgs
        if not self.solve_result.solveSucceeded:
            self.logger.debug("Solve failed.")
            return
        self.logger.debug("Solve succeeded.")
        self.job_result["solveSucceeded"] = True

        # Save output
        # Example: ODLines_O_1_1000_D_2001_3000.csv
        out_filename = (f"ODLines_O_{origins_criteria[0]}_{origins_criteria[1]}_"
                        f"D_{destinations_criteria[0]}_{destinations_criteria[1]}")
        if self.output_format is helpers.OutputFormat.featureclass:
            self._export_to_feature_class(out_filename)
        elif self.output_format is helpers.OutputFormat.csv:
            out_csv_file = os.path.join(self.output_od_location, f"{out_filename}.csv")
            self._export_to_csv(out_csv_file)
        elif self.output_format is helpers.OutputFormat.arrow:
            out_arrow_file = os.path.join(self.output_od_location, f"{out_filename}.arrow")
            self._export_to_arrow(out_arrow_file)

        self.logger.debug("Finished calculating Route.")

    def _export_to_feature_class(self, out_fc_name):
        """Export the OD Lines result to a feature class."""
        # Make output gdb
        self.logger.debug("Creating output geodatabase for Route results...")
        od_workspace = os.path.join(self.job_folder, "scratch.gdb")
        run_gp_tool(
            arcpy.management.CreateFileGDB,
            [os.path.dirname(od_workspace), os.path.basename(od_workspace)],
            log_to_use=self.logger
        )

        output_od_lines = os.path.join(od_workspace, out_fc_name)
        self.logger.debug(f"Exporting Route Lines output to {output_od_lines}...")
        self.solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, output_od_lines)
        self.job_result["outputLines"] = output_od_lines

        # For services solve, export Origins and Destinations and properly populate OriginOID and DestinationOID fields
        # in the output Lines. Services do not preserve the original input OIDs, instead resetting from 1, unlike solves
        # using a local network dataset, so this extra post-processing step is necessary.
        if self.is_service:
            output_origins = os.path.join(od_workspace, "output_od_origins")
            self.logger.debug(f"Exporting Route Origins output to {output_origins}...")
            self.solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Origins, output_origins)
            output_destinations = os.path.join(od_workspace, "output_od_destinations")
            self.logger.debug(f"Exporting Route Destinations output to {output_destinations}...")
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
                 f"!{os.path.basename(output_origins)}.{self.orig_origin_oid_field}!", "PYTHON3"]
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
                 f"!{os.path.basename(output_destinations)}.{self.orig_dest_oid_field}!", "PYTHON3"]
            )
            run_gp_tool(arcpy.management.RemoveJoin, [lines_layer_name])

    def _export_to_csv(self, out_csv_file):
        """Save the OD Lines result to a CSV file."""
        self.logger.debug(f"Saving Route Lines output to CSV as {out_csv_file}.")

        # For services solve, properly populate OriginOID and DestinationOID fields in the output Lines. Services do
        # not preserve the original input OIDs, instead resetting from 1, unlike solves using a local network dataset,
        # so this extra post-processing step is necessary.
        if self.is_service:
            # Read the Lines output
            with self.solve_result.searchCursor(
                arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, self.output_fields
            ) as cur:
                od_df = pd.DataFrame(cur, columns=self.output_fields)
            # Read the Origins output and transfer original OriginOID to Lines
            origins_columns = ["ObjectID", self.orig_origin_oid_field]
            with self.solve_result.searchCursor(
                arcpy.nax.OriginDestinationCostMatrixOutputDataType.Origins, origins_columns
            ) as cur:
                origins_df = pd.DataFrame(cur, columns=origins_columns)
            origins_df.set_index("ObjectID", inplace=True)
            od_df = od_df.join(origins_df, "OriginOID")
            del origins_df
            # Read the Destinations output and transfer original DestinationOID to Lines
            dest_columns = ["ObjectID", self.orig_dest_oid_field]
            with self.solve_result.searchCursor(
                arcpy.nax.OriginDestinationCostMatrixOutputDataType.Destinations, dest_columns
            ) as cur:
                dests_df = pd.DataFrame(cur, columns=dest_columns)
            dests_df.set_index("ObjectID", inplace=True)
            od_df = od_df.join(dests_df, "DestinationOID")
            del dests_df
            # Clean up and rename columns
            od_df.drop(["OriginOID", "DestinationOID"], axis="columns", inplace=True)
            od_df.rename(
                columns={self.orig_origin_oid_field: "OriginOID", self.orig_dest_oid_field: "DestinationOID"},
                inplace=True
            )
            # Write CSV file
            od_df.to_csv(out_csv_file, index=False)

        else:  # Local network dataset output
            with open(out_csv_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.output_fields)
                for row in self.solve_result.searchCursor(
                    arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines,
                    self.output_fields
                ):
                    writer.writerow(row)

        self.job_result["outputLines"] = out_csv_file

    def _select_inputs(self, origins_criteria):
        """Create layers from the origins so the layer contains only the desired inputs for the chunk.

        Args:
            origins_criteria (list): Origin ObjectID range to select from the input dataset
        """
        # Select the origins with ObjectIDs in this range
        self.logger.debug("Selecting origins for this chunk...")
        origins_where_clause = (
            f"{self.origins_oid_field_name} >= {origins_criteria[0]} "
            f"And {self.origins_oid_field_name} <= {origins_criteria[1]}"
        )
        self.logger.debug(f"Origins where clause: {origins_where_clause}")
        self.input_origins_layer_obj = run_gp_tool(
            arcpy.management.MakeFeatureLayer,
            [self.origins, self.input_origins_layer, origins_where_clause],
            log_to_use=self.logger
        ).getOutput(0)
        num_origins = int(arcpy.management.GetCount(self.input_origins_layer_obj).getOutput(0))
        self.logger.debug(f"Number of origins selected: {num_origins}")

    def _insert_stops(self):
        """Insert the origins and destinations as stops for the analysis."""
        location_fields = ["SourceID", "SourceOID", "PosAlong", "SideOfEdge"]
        ## TODO: Only do loc fields when relevant
        # Make a layer for destinations for quicker access
        self.input_origins_layer_obj = run_gp_tool(
            arcpy.management.MakeFeatureLayer,
            [self.destinations, self.input_destinations_layer],
            log_to_use=self.logger
        )
        # Store a dictionary of destinations used by this group of origins for quick lookups
        destinations = {}
        # Use an insertCursor to insert Stops into the Route analysis
        with self.rt_solver.insertCursor(
            arcpy.nax.RouteInputDataType.Stops,
            ["RouteName", "Sequence", "SHAPE@", "Name"] + location_fields
        ) as icur:
            # Loop through origins and insert them into Stops along with their assigned destinations
            ## TODO: Deal with Name field
            for origin_row in arcpy.da.SearchCursor(
                self.input_origins_layer_obj,
                ["SHAPE@", "ObjectID", self.assigned_dest_field]
            ):
                dest_id = origin_row[1]
                if dest_id not in destinations:
                    with arcpy.da.SearchCursor(
                        self.input_destinations_layer,
                        ["SHAPE@", self.dest_id_field] + location_fields,
                        where=f"{self.dest_id_field} = {dest_id}"  ## TODO: Update for str/int
                    ) as cur:
                        try:
                            destinations[dest_id] = next(cur)
                        except StopIteration:
                            # The origin's destination is not present in the destinations table. Just skip the origin.
                            continue
            # Insert origin and destination
            destination_row = destinations[dest_id]
            route_name = f"{origin_row[1]} - {dest_id}"
            icur.insertRow([route_name, 1, origin_row[0], origin_row[1], None, None, None, None])
            icur.insertRow([route_name, 2] + list(destination_row))

    def _determine_if_travel_mode_time_based(self):
        """Determine if the travel mode uses a time-based impedance attribute."""
        # Get the travel mode object from the already-instantiated OD solver object. This saves us from having to parse
        # the user's input travel mode from its string name, object, or json representation.
        travel_mode = self.rt_solver.travelMode
        impedance = travel_mode.impedance
        time_attribute = travel_mode.timeAttributeName
        distance_attribute = travel_mode.distanceAttributeName
        self.is_travel_mode_time_based = time_attribute == impedance
        self.is_travel_mode_dist_based = distance_attribute == impedance
        # Determine which of the OD Lines output table fields contains the optimized cost values
        if not self.is_travel_mode_time_based and not self.is_travel_mode_dist_based:
            self.optimized_field_name = "Total_Other"
            self.output_fields.append(self.optimized_field_name)
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


def solve_route(inputs, chunk):
    """Solve an Route analysis for the given inputs for the given chunk of ObjectIDs.

    Args:
        inputs (dict): Dictionary of keyword inputs suitable for initializing the ODCostMatrix class
        chunk (list): Represents the ObjectID ranges to select from the origins and destinations when solving the OD
            Cost Matrix. For example, [[1, 1000], [4001, 5000]] means use origin OIDs 1-1000 and destination OIDs
            4001-5000.

    Returns:
        dict: Dictionary of results from the ODCostMatrix class
    """
    rt = ODCostMatrix(**inputs)
    rt.logger.info((
        f"Processing origins OID {chunk[0][0]} to {chunk[0][1]} and destinations OID {chunk[1][0]} to {chunk[1][1]} "
        f"as job id {rt.job_id}"
    ))
    rt.solve(chunk[0], chunk[1])
    return rt.job_result


class ParallelODCalculator:
    """Solves a large Route by chunking the problem, solving in parallel, and combining results."""

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, origins, assigned_dest_field, destinations, dest_id_field, network_data_source, travel_mode,
        max_routes, max_processes, time_units, distance_units,
        time_of_day=None, barriers=None
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
            barriers (list(str), optional): List of catalog paths to point, line, and polygon barriers to use.
                Defaults to None.
        """
        self.origins = origins
        self.assigned_dest_field = assigned_dest_field
        self.destinations = destinations
        self.dest_id_field = dest_id_field
        self.max_routes = max_routes
        time_units = helpers.convert_time_units_str_to_enum(time_units)
        distance_units = helpers.convert_distance_units_str_to_enum(distance_units)
        if not barriers:
            barriers = []
        self.max_processes = max_processes
        if not time_of_day:
            self.time_of_day = None
        else:
            self.time_of_day = datetime.datetime.strptime(time_of_day, helpers.DATETIME_FORMAT)

        # Scratch folder to store intermediate outputs from the Route processes
        unique_id = uuid.uuid4().hex
        self.scratch_folder = os.path.join(arcpy.env.scratchFolder, "rt_" + unique_id)  # pylint: disable=no-member
        LOGGER.info(f"Intermediate outputs will be written to {self.scratch_folder}.")
        os.mkdir(self.scratch_folder)

        # Initialize the dictionary of inputs to send to each OD solve
        self.rt_inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "output_format": self.output_format,
            "output_od_location": self.output_od_location,
            "network_data_source": network_data_source,
            "travel_mode": travel_mode,
            "scratch_folder": self.scratch_folder,
            "time_units": time_units,
            "distance_units": distance_units,
            "time_of_day": self.time_of_day,
            "barriers": barriers
        }

        # List of intermediate output OD Line files created by each process
        self.od_line_files = []

        # Construct OID ranges for chunks of origins and destinations
        self.origin_ranges = self._get_oid_ranges_for_stops()

        # Calculate the total number of jobs to use in logging
        self.total_jobs = len(self.origin_ranges)

        self.optimized_cost_field = None

    def _validate_route_settings(self):
        """Validate Route settings before spinning up a bunch of parallel processes doomed to failure.

        Also check which field name in the output OD Lines will store the optimized cost values. This depends on the
        travel mode being used by the analysis, and we capture it here to use in later steps.

        Returns:
            str: The name of the field in the output OD Lines table containing the optimized costs for the analysis
        """
        # Create a dummy ODCostMatrix object, initialize an OD solver object, and set properties. This allows us to
        # detect any errors prior to spinning up a bunch of parallel processes and having them all fail.
        LOGGER.debug("Validating Route settings...")
        optimized_cost_field = None
        rt = None
        try:
            rt = Route(**self.rt_inputs)
            rt.initialize_od_solver()
            # Check which field name in the output OD Lines will store the optimized cost values
            optimized_cost_field = rt.optimized_field_name
            LOGGER.debug("Route settings successfully validated.")
        except Exception:
            LOGGER.error("Error initializing Route analysis.")
            errs = traceback.format_exc().splitlines()
            for err in errs:
                LOGGER.error(err)
            raise
        finally:
            if rt:
                LOGGER.debug("Deleting temporary test Route job folder...")
                # Close logging
                for handler in rt.logger.handlers:
                    handler.close()
                    rt.logger.removeHandler(handler)
                # Delete output folder
                shutil.rmtree(rt.job_result["jobFolder"], ignore_errors=True)
                del rt

        return optimized_cost_field

    def _get_oid_ranges_for_stops(self):
        """Construct ranges of ObjectIDs for use in where clauses to split large data into chunks.

        The origins table should already be sorted by the assigned destination field for best efficiency.

        Returns:
            list: list of ObjectID ranges for the current dataset representing each chunk. For example,
                [[1, 1000], [1001, 2000], [2001, 2478]] represents three chunks of no more than 1000 rows.
        """
        ranges = []
        num_in_range = 0
        current_range = [0, 0]
        # Loop through all OIDs of the origins and construct tuples of min and max OID for each chunk
        # We do it this way and not by straight-up looking at the numerical values of OIDs to account
        # for definition queries, selection sets, or feature layers with gaps in OIDs
        for row in arcpy.da.SearchCursor(self.origins, ["OID@"]):  # pylint: disable=no-member
            oid = row[0]
            if num_in_range == 0:
                # Starting new range
                current_range[0] = oid
            # Increase the count of items in this range and set the top end of the range to the current oid
            num_in_range += 1
            current_range[1] = oid
            if num_in_range == self.max_routes:
                # Finishing up a chunk
                ranges.append(current_range)
                # Reset range trackers
                num_in_range = 0
                current_range = [0, 0]
        # After looping, close out the last range if we still have one open
        if current_range != [0, 0]:
            ranges.append(current_range)

        return ranges

    def solve_route_in_parallel(self):
        """Solve the Route in chunks and post-process the results."""
        # Validate Route settings. Essentially, create a dummy Route class instance and set up the
        # solver object to ensure this at least works. Do this up front before spinning up a bunch of parallel processes
        # the optimized that are guaranteed to all fail. While we're doing this, check and store the field name that
        # will represent costs in the output OD Lines table. We'll use this in post processing.
        ## TODO
        self.optimized_cost_field = self._validate_route_settings()

        # Compute Route in parallel
        completed_jobs = 0  # Track the number of jobs completed so far to use in logging
        # Use the concurrent.futures ProcessPoolExecutor to spin up parallel processes that solve the OD cost matrices
        with futures.ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            # Each parallel process calls the solve_od_cost_matrix() function with the od_inputs dictionary for the
            # given origin and destination OID ranges.
            jobs = {executor.submit(solve_route, self.rt_inputs, range): range for range in self.origin_ranges}
            # As each job is completed, add some logging information and store the results to post-process later
            for future in futures.as_completed(jobs):
                completed_jobs += 1
                LOGGER.info(
                    f"Finished Route calculation {completed_jobs} of {self.total_jobs}.")
                try:
                    # The Route job returns a results dictionary. Retrieve it.
                    result = future.result()
                except Exception:
                    # If we couldn't retrieve the result, some terrible error happened. Log it.
                    LOGGER.error("Failed to get Route result from parallel processing.")
                    errs = traceback.format_exc().splitlines()
                    for err in errs:
                        LOGGER.error(err)
                    raise

                # Parse the results dictionary and store components for post-processing.
                if result["solveSucceeded"]:
                    self.od_line_files.append(result["outputLines"])
                else:
                    # Typically, a solve fails because no destinations were found for any of the origins in the chunk,
                    # and this is a perfectly legitimate failure. It is not an error. However, they may be other, less
                    # likely, reasons for solve failure. Write solve messages to the main GP message thread in debug
                    # mode only in case the user is having problems. The user can also check the individual OD log
                    # files.
                    LOGGER.debug(f"Solve failed for job id {result['jobId']}.")
                    LOGGER.debug(result["solveMessages"])

        # Post-process outputs
        if self.od_line_files:
            LOGGER.info("Post-processing Route results...")
            self.od_line_files = sorted(self.od_line_files)
            if self.output_format is helpers.OutputFormat.featureclass:
                self._post_process_od_line_fcs()
            elif self.output_format is helpers.OutputFormat.csv:
                self._post_process_od_line_csvs()
            elif self.output_format is helpers.OutputFormat.arrow:
                self._post_process_od_line_arrow_files()
        else:
            LOGGER.warning("All Route solves failed, so no output was produced.")

        # Clean up
        # Delete the job folders if the job succeeded
        if DELETE_INTERMEDIATE_OD_OUTPUTS:
            LOGGER.info("Deleting intermediate outputs...")
            try:
                shutil.rmtree(self.scratch_folder, ignore_errors=True)
            except Exception:  # pylint: disable=broad-except
                # If deletion doesn't work, just throw a warning and move on. This does not need to kill the tool.
                LOGGER.warning(f"Unable to delete intermediate Route output folder {self.scratch_folder}.")

        LOGGER.info("Finished calculating OD Cost Matrices.")

    def _post_process_od_line_fcs(self):
        """Merge and post-process the OD Lines calculated in each separate process.

        Create an empty final output feature class and populate it from each of the intermediate OD Lines feature
        classes. Do this instead of simply using the Merge tool in order to correctly calculate the DestinationRank
        field and eliminate extra records when only the k closest destinations should be found.

        For the case where we wanted to find only the k closest destinations for each origin, calculating the OD in
        chunks means our combined output may have more than k destinations for each origin because each individual chunk
        found the closest k for that chunk. We need to eliminate all extra rows beyond the first k.

        Calculating the OD in chunks also means the DestinationRank field calculated by each chunk is not correct for
        the entire analysis. DestinationRank refers to the rank within the chunk, not the overall rank. We need to
        recalculate DestinationRank considering the entire dataset.
        """
        # Create the final output feature class
        desc = arcpy.Describe(self.od_line_files[0])
        run_gp_tool(arcpy.management.CreateFeatureclass, [
            os.path.dirname(self.output_od_location),
            os.path.basename(self.output_od_location),
            "POLYLINE",
            self.od_line_files[0],  # template feature class to transfer full schema
            "SAME_AS_TEMPLATE",
            "SAME_AS_TEMPLATE",
            desc.spatialReference
        ])

        # Insert the rows from all the individual output feature classes into the final output
        fields = ["SHAPE@"] + [f.name for f in desc.fields]
        with arcpy.da.InsertCursor(self.output_od_location, fields) as cur:  # pylint: disable=no-member
            # Handle each origin range separately to avoid pulling all results into memory at once
            for origin_range in self.origin_ranges:
                fcs_for_origin_range = [
                    f for f in self.od_line_files if os.path.basename(f).startswith(
                        f"ODLines_O_{origin_range[0]}_{origin_range[1]}_"
                    )
                ]
                if not fcs_for_origin_range:
                    # No records for this chunk of origins. Just move on to the next chunk of origins.
                    continue
                if len(fcs_for_origin_range) < 2:
                    # All results for this chunk of origins are already in the same table, so
                    # there is no need to post-process because the k closest have already been found, and the
                    # DestinationRank field is already correct. Just insert the records directly into the final output
                    # table.
                    for row in arcpy.da.SearchCursor(fcs_for_origin_range[0], fields):  # pylint: disable=no-member
                        cur.insertRow(row)
                    continue
                # If there are multiple feature classes to handle at once, we need to eliminate extra rows and
                # properly update the DestinationRank field. To do this, read them into pandas.
                fc_dfs = []
                df_fields = ["Shape"] + fields[1:]
                for fc in fcs_for_origin_range:
                    with arcpy.da.SearchCursor(fc, fields) as cur2:  # pylint: disable=no-member
                        fc_dfs.append(pd.DataFrame(cur2, columns=df_fields))
                df = pd.concat(fc_dfs, ignore_index=True)
                # Drop all but the k nearest rows for each OriginOID (if needed) and calculate DestinationRank
                df = self._update_df_for_k_nearest_and_destination_rank(df)
                # Write the pandas rows to the final output feature class
                for row in df.itertuples(index=False, name=None):
                    cur.insertRow(row)

        LOGGER.info("Post-processing complete.")
        LOGGER.info(f"Results written to {self.output_od_location}.")

    def _post_process_od_line_csvs(self):
        """Post-process CSV file outputs."""
        # If we wanted to find only the k closest destinations for each origin, we have to do additional post-
        # processing. Calculating the OD in chunks means our merged output may have more than k destinations for each
        # origin because each individual chunk found the closest k for that chunk. We need to eliminate all extra rows
        # beyond the first k. Sort the data by OriginOID and the Total_ field that was optimized for the analysis.
        if self.num_destinations:
            # Handle each origin range separately to avoid pulling all results into memory at once
            for origin_range in self.origin_ranges:
                csvs_for_origin_range = [
                    f for f in self.od_line_files if os.path.basename(f).startswith(
                        f"ODLines_O_{origin_range[0]}_{origin_range[1]}_"
                    )
                ]
                if len(csvs_for_origin_range) < 2:
                    # Either there were no results for this chunk, or all results are already in the same table, so
                    # there is no need to post-process because the k closest have already been found.
                    continue

                # Read the csv files into a pandas dataframe for easy sorting
                df = pd.concat(map(pd.read_csv, csvs_for_origin_range), ignore_index=True)
                # Drop all but the k nearest rows for each OriginOID and calculate DestinationRank
                df = self._update_df_for_k_nearest_and_destination_rank(df)

                # Write the updated CSV file and delete the originals
                out_csv = os.path.join(self.output_od_location, f"ODLines_O_{origin_range[0]}_{origin_range[1]}.csv")
                df.to_csv(out_csv, index=False)
                for csv_file in csvs_for_origin_range:
                    os.remove(csv_file)

    def _post_process_od_line_arrow_files(self):
        """Post-process Arrow file outputs."""
        # If we wanted to find only the k closest destinations for each origin, we have to do additional post-
        # processing. Calculating the OD in chunks means our merged output may have more than k destinations for each
        # origin because each individual chunk found the closest k for that chunk. We need to eliminate all extra rows
        # beyond the first k. Sort the data by OriginOID and the Total_ field that was optimized for the analysis.
        if self.num_destinations:
            # Handle each origin range separately to avoid pulling all results into memory at once
            for origin_range in self.origin_ranges:
                files_for_origin_range = [
                    f for f in self.od_line_files if os.path.basename(f).startswith(
                        f"ODLines_O_{origin_range[0]}_{origin_range[1]}_"
                    )
                ]
                if len(files_for_origin_range) < 2:
                    # Either there were no results for this chunk, or all results are already in the same table, so
                    # there is no need to post-process because the k closest have already been found.
                    continue

                # Read the Arrow files into a pandas dataframe for easy sorting
                # Note: An Arrow dataset could reasonably be used here, along with native Arrow table
                # manipulations, instead of reading the files into pandas dataframes. However, the pyarrow
                # version included in ArcGIS Pro (as of 2.9) does not support the new dataset option and
                # other useful newer Arrow features. Additionally, using pandas allows us to share code
                # with the CSV post-processing.
                arrow_dfs = []
                for arrow_file in files_for_origin_range:
                    with pa.memory_map(arrow_file, 'r') as source:
                        batch_reader = pa.ipc.RecordBatchFileReader(source)
                        arrow_dfs.append(batch_reader.read_all().to_pandas(split_blocks=True))
                df = pd.concat(arrow_dfs, ignore_index=True)

                # Drop all but the k nearest rows for each OriginOID and calculate DestinationRank
                df = self._update_df_for_k_nearest_and_destination_rank(df)

                # Write the updated Arrow file and delete the originals
                table = pa.Table.from_pandas(df)
                out_file = os.path.join(self.output_od_location, f"ODLines_O_{origin_range[0]}_{origin_range[1]}.arrow")
                local = fs.LocalFileSystem()
                with local.open_output_stream(out_file) as f:
                    with pa.RecordBatchFileWriter(f, table.schema) as writer:
                        writer.write_table(table)
                del df
                del table
                del arrow_dfs
                del batch_reader
                for arrow_file in files_for_origin_range:
                    os.remove(arrow_file)

    def _update_df_for_k_nearest_and_destination_rank(self, df):
        """Drop all but the k nearest records for each Origin from the dataframe and calculate DestinationRank."""
        # Sort according to OriginOID and cost field
        df.sort_values(["OriginOID", self.optimized_cost_field], inplace=True)
        # Keep only the first k records for each OriginOID
        if self.num_destinations:
            df = df.groupby("OriginOID").head(self.num_destinations).reset_index(drop=True)
        # Properly calculate the DestinationRank field
        df["DestinationRank"] = df.groupby("OriginOID").cumcount() + 1
        return df


def launch_parallel_od():
    """Read arguments passed in via subprocess and run the parallel Route.

    This script is intended to be called via subprocess via the solve_large_rt.py module, which does essential
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
    help_string = ("The desired format for the output Route Lines results. "
                   f"Choices: {', '.join(helpers.OUTPUT_FORMATS)}")
    parser.add_argument(
        "-of", "--output-format", action="store", dest="output_format", help=help_string, required=True)

    # ----output-od-location parameter
    help_string = "The catalog path to the output feature class or folder that will contain the Route results."
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
        "Maximum number of origins that can be in one chunk for parallel processing of Route solves. "
        "For example, 1000 means that a chunk consists of no more than 1000 origins and max-destination destinations."
    )
    parser.add_argument(
        "-mo", "--max-origins", action="store", dest="max_origins", type=int, help=help_string, required=True)

    # --max-destinations parameter
    help_string = (
        "Maximum number of destinations that can be in one chunk for parallel processing of Route solves. "
        "For example, 1000 means that a chunk consists of no more than max-origin origins and 1000 destinations."
    )
    parser.add_argument(
        "-md", "--max-destinations", action="store", dest="max_destinations", type=int, help=help_string, required=True)

    # --max-processes parameter
    help_string = "Maximum number parallel processes to use for the Route solves."
    parser.add_argument(
        "-mp", "--max-processes", action="store", dest="max_processes", type=int, help=help_string, required=True)

    # --cutoff parameter
    help_string = (
        "Impedance cutoff to limit the Route search distance. Should be specified in the same units as the "
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
    parser.add_argument("-tod", "--time-of-day", action="store", dest="time_of_day", help=help_string,required=False)

    # --barriers parameter
    help_string = "A list of catalog paths to the feature classes containing barriers to use in the Route."
    parser.add_argument(
        "-b", "--barriers", action="store", dest="barriers", help=help_string, nargs='*', required=False)

    # Get arguments as dictionary.
    args = vars(parser.parse_args())

    # Initialize a parallel Route calculator class
    od_calculator = ParallelODCalculator(**args)
    # Solve the Route in parallel chunks
    start_time = time.time()
    od_calculator.solve_od_in_parallel()
    LOGGER.info(f"Parallel Route calculation completed in {round((time.time() - start_time) / 60, 2)} minutes")


if __name__ == "__main__":
    # This script should always be launched via subprocess as if it were being called from the command line.
    launch_parallel_od()
