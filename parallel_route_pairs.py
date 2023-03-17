"""Compute a large multi-route analysis by chunking the preassigned origin-destination pairs and solving in parallel.

Multiple cases are supported:
- one-to-one: A field in the input origins table indicates which destination the origin is assigned to
- many-to-many: A separate table defines a list of origin-destination pairs. A single origin may be assigned to multiple
    destinations.

The outputs are written to a single combined feature class.

This is a sample script users can modify to fit their specific needs.

This script is intended to be called as a subprocess from the solve_large_route_pair_analysis.py script so that it can
launch parallel processes with concurrent.futures. It must be called as a subprocess because the main script tool
process, when running within ArcGIS Pro, cannot launch parallel subprocesses on its own.

This script should not be called directly from the command line.

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
# pylint: disable=logging-fstring-interpolation
from concurrent import futures
import os
import sys
import uuid
import logging
import shutil
import time
import datetime
import traceback
import argparse
from math import ceil
from distutils.util import strtobool
import pandas as pd

import arcpy

# Import Route settings from config file
from rt_config import RT_PROPS, RT_PROPS_SET_BY_TOOL

import helpers

arcpy.env.overwriteOutput = True

# Set logging for the main process.
# LOGGER logs everything from the main process to stdout using a specific format that the SolveLargeRoute tool
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


class Route:  # pylint:disable = too-many-instance-attributes
    """Used for solving a Route problem in parallel for a designated chunk of the input datasets."""

    def __init__(self, **kwargs):
        """Initialize the Route analysis for the given inputs.

        Expected arguments:
        - pair_type
        - origins
        - origin_id_field
        - destinations
        - dest_id_field
        - network_data_source
        - travel_mode
        - time_units
        - distance_units
        - time_of_day
        - reverse_direction
        - scratch_folder
        - assigned_dest_field
        - od_pair_table
        - origin_transfer_fields
        - destination_transfer_fields
        - barriers
        """
        self.pair_type = kwargs["pair_type"]
        self.origins = kwargs["origins"]
        self.origin_id_field = kwargs["origin_id_field"]
        self.destinations = kwargs["destinations"]
        self.dest_id_field = kwargs["dest_id_field"]
        self.network_data_source = kwargs["network_data_source"]
        self.travel_mode = kwargs["travel_mode"]
        self.time_units = kwargs["time_units"]
        self.distance_units = kwargs["distance_units"]
        self.time_of_day = kwargs["time_of_day"]
        self.reverse_direction = kwargs["reverse_direction"]
        self.scratch_folder = kwargs["scratch_folder"]
        self.assigned_dest_field = kwargs["assigned_dest_field"]
        self.od_pair_table = kwargs["od_pair_table"]
        self.origin_transfer_fields = kwargs["origin_transfer_fields"]
        self.destination_transfer_fields = kwargs["destination_transfer_fields"]
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

        # Get field objects for the origin and destination ID fields since we need this in multiple places
        self.origin_id_field_obj = arcpy.ListFields(self.origins, wild_card=self.origin_id_field)[0]
        self.dest_id_field_obj = arcpy.ListFields(self.destinations, wild_card=self.dest_id_field)[0]

        # Set up other instance attributes
        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.rt_solver = None
        self.solve_result = None
        self.input_origins_layer = "InputOrigins" + self.job_id
        self.input_destinations_layer = "InputDestinations" + self.job_id
        self.input_origins_layer_obj = None
        self.input_dests_layer_obj = None
        self.origin_unique_id_field_name = "OriginUniqueID"
        self.dest_unique_id_field_name = "DestinationUniqueID"
        self.od_pairs = None

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
            helpers.run_gp_tool(
                self.logger,
                arcpy.na.MakeNetworkDatasetLayer,
                [self.network_data_source, nds_layer_name],
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
                # Suppress warnings for older services (pre 11.0) that don't support locate settings and services
                # that don't support accumulating attributes because we don't want the tool to always throw a warning.
                if not (self.is_service and prop in [
                    "searchTolerance", "searchToleranceUnits", "accumulateAttributeNames"
                ]):
                    self.logger.warning(
                        f"Failed to set property {prop} from OD config file. Default will be used instead.")
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

    def _add_unique_id_fields(self):
        """Add fields to input Stops with the origin and destination's original unique IDs."""
        field_types = {"String": "TEXT", "Single": "FLOAT", "Double": "DOUBLE", "SmallInteger": "SHORT",
                       "Integer": "LONG", "OID": "LONG"}
        origin_field_def = [self.origin_unique_id_field_name, field_types[self.origin_id_field_obj.type]]
        if self.origin_id_field_obj.type == "String":
            origin_field_def += [self.origin_unique_id_field_name, self.origin_id_field_obj.length]
        dest_field_def = [self.dest_unique_id_field_name, field_types[self.dest_id_field_obj.type]]
        if self.dest_id_field_obj.type == "String":
            dest_field_def += [self.dest_unique_id_field_name, self.dest_id_field_obj.length]
        self.rt_solver.addFields(arcpy.nax.RouteInputDataType.Stops, [origin_field_def, dest_field_def])

    def _select_inputs_one_to_one(self, origins_criteria):
        """Create layers from the origins so the layer contains only the desired inputs for the chunk.

        Args:
            origins_criteria (list): Origin ObjectID range to select from the input dataset
        """
        # Select the origins with ObjectIDs in this range
        self.logger.debug("Selecting origins for this chunk...")
        origins_oid_field_name = arcpy.Describe(self.origins).oidFieldName
        origins_where_clause = (
            f"{origins_oid_field_name} >= {origins_criteria[0]} "
            f"And {origins_oid_field_name} <= {origins_criteria[1]}"
        )
        self.logger.debug(f"Origins where clause: {origins_where_clause}")
        self.input_origins_layer_obj = helpers.run_gp_tool(
            self.logger,
            arcpy.management.MakeFeatureLayer,
            [self.origins, self.input_origins_layer, origins_where_clause],
        ).getOutput(0)
        num_origins = int(arcpy.management.GetCount(self.input_origins_layer_obj).getOutput(0))
        self.logger.debug(f"Number of origins selected: {num_origins}")

        # Make a layer for destinations for quicker access
        helpers.run_gp_tool(
            self.logger,
            arcpy.management.MakeFeatureLayer,
            [self.destinations, self.input_destinations_layer],
        )

    def _get_od_pairs_for_chunk(self, chunk_definition):
        """Retrieve a list of OD pairs included in this chunk.

        Args:
            chunk_definition (list): A list of [chunk starting row number, chunk size] indicating which records to
                retrieve from the OD pair table.
        """
        # Read the relevant rows from the CSV
        chunk_num, chunk_size = chunk_definition
        # Explicitly set data types
        dtypes = {
            0: helpers.PD_FIELD_TYPES[self.origin_id_field_obj.type],
            1: helpers.PD_FIELD_TYPES[self.dest_id_field_obj.type]
        }
        df_od_pairs = pd.read_csv(
            self.od_pair_table,
            header=None,
            skiprows=chunk_size*chunk_num,
            nrows=chunk_size,
            dtype=dtypes
        )
        self.od_pairs = df_od_pairs.values.tolist()

    def _select_inputs_many_to_many(self):
        """Create layers that include only the origins and destinations relevant to this chunk."""
        # Select the origins present in this chunk of predefined OD pairs
        self.logger.debug("Selecting origins for this chunk...")
        origins_in_chunk = set([pair[0] for pair in self.od_pairs])
        if isinstance(self.od_pairs[0][0], (int, float,)):
            origin_string = ", ".join([str(o_id) for o_id in origins_in_chunk])
        else:
            origin_string = "'" + "', '".join([str(o_id) for o_id in origins_in_chunk]) + "'"
        origins_where_clause = f"{self.origin_id_field} IN ({origin_string})"
        self.logger.debug(f"Origins where clause: {origins_where_clause}")
        self.input_origins_layer_obj = helpers.run_gp_tool(
            self.logger,
            arcpy.management.MakeFeatureLayer,
            [self.origins, self.input_origins_layer, origins_where_clause]
        ).getOutput(0)
        num_origins = int(arcpy.management.GetCount(self.input_origins_layer).getOutput(0))
        self.logger.debug(f"Number of origins selected: {num_origins}")
        # Select the destinations present in this chunk of predefined OD pairs
        self.logger.debug("Selecting destinations for this chunk...")
        dests_in_chunk = set([pair[1] for pair in self.od_pairs])
        if isinstance(self.od_pairs[0][1], (int, float,)):
            dest_string = ", ".join([str(d_id) for d_id in dests_in_chunk])
        else:
            dest_string = "'" + "', '".join([str(d_id) for d_id in dests_in_chunk]) + "'"
        dests_where_clause = f"{self.dest_id_field} IN ({dest_string})"
        self.logger.debug(f"Destinations where clause: {dests_where_clause}")
        self.input_dests_layer_obj = helpers.run_gp_tool(
            self.logger,
            arcpy.management.MakeFeatureLayer,
            [self.destinations, self.input_destinations_layer, dests_where_clause]
        ).getOutput(0)
        num_dests = int(arcpy.management.GetCount(self.input_destinations_layer).getOutput(0))
        self.logger.debug(f"Number of destinations selected: {num_dests}")

    def _insert_stops_one_to_one(self):  # pylint: disable=too-many-locals
        """Insert the origins and destinations as Stops for the Route analysis for the one-to-one case."""
        # Use an insertCursor to insert Stops into the Route analysis
        destinations = {}
        destination_rows = []
        with self.rt_solver.insertCursor(
            arcpy.nax.RouteInputDataType.Stops,
            ["RouteName", "Sequence", self.origin_unique_id_field_name, "SHAPE@", self.dest_unique_id_field_name] +
                self.origin_transfer_fields
        ) as icur:
            # Loop through origins and insert them into Stops along with their assigned destinations
            for origin in arcpy.da.SearchCursor(  # pylint: disable=no-member
                self.input_origins_layer,
                ["SHAPE@", self.origin_id_field, self.assigned_dest_field] + self.origin_transfer_fields
            ):
                dest_id = origin[2]
                if dest_id is None:
                    continue
                if dest_id not in destinations:
                    dest_val = f"'{dest_id}'" if isinstance(dest_id, str) else dest_id
                    with arcpy.da.SearchCursor(  # pylint: disable=no-member
                        self.input_destinations_layer,
                        ["SHAPE@", self.dest_id_field] + self.destination_transfer_fields,
                        where_clause=f"{self.dest_id_field} = {dest_val}"
                    ) as cur:
                        try:
                            destinations[dest_id] = next(cur)
                        except StopIteration:
                            # The origin's destination is not present in the destinations table. Just skip the origin.
                            continue
                # Insert origin and destination
                destination = destinations[dest_id]
                if self.reverse_direction:
                    route_name = f"{dest_id} - {origin[1]}"
                    origin_sequence = 2
                    destination_sequence = 1
                else:
                    route_name = f"{origin[1]} - {dest_id}"
                    origin_sequence = 1
                    destination_sequence = 2
                # Define the final origin and destination rows for the input Stops
                origin_row = [route_name, origin_sequence, origin[1], origin[0], None] + list(origin)[3:]
                destination_row = [route_name, destination_sequence, None, destination[0], destination[1]] + \
                    list(destination)[2:]
                icur.insertRow(origin_row)
                destination_rows.append(destination_row)

        # Insert destinations
        with self.rt_solver.insertCursor(
            arcpy.nax.RouteInputDataType.Stops,
            ["RouteName", "Sequence", self.origin_unique_id_field_name, "SHAPE@", self.dest_unique_id_field_name] +
                self.destination_transfer_fields
        ) as dcur:
            for row in destination_rows:
                dcur.insertRow(row)

    def _insert_stops_many_to_many(self):
        """Insert each predefined OD pair into the Route analysis for the many-to-many case."""
        # Store data of the relevant origins and destinations in dictionaries for quick lookups and reuse
        o_data = {}  # {Origin ID: [Shape, transferred fields]}
        for row in arcpy.da.SearchCursor(  # pylint: disable=no-member
            self.input_origins_layer,
            [self.origin_id_field, "SHAPE@"] + self.origin_transfer_fields
        ):
            o_data[row[0]] = row[1:]
        d_data = {}  # {Destination ID: [Shape, transferred fields]}
        for row in arcpy.da.SearchCursor(  # pylint: disable=no-member
            self.input_destinations_layer,
            [self.dest_id_field, "SHAPE@"] + self.destination_transfer_fields
        ):
            d_data[row[0]] = row[1:]

        # Insert origins from each OD pair into the Route analysis
        with self.rt_solver.insertCursor(
            arcpy.nax.RouteInputDataType.Stops,
            ["RouteName", "Sequence", self.origin_unique_id_field_name, "SHAPE@"] + self.origin_transfer_fields
        ) as icur:
            for od_pair in self.od_pairs:
                origin_id, dest_id = od_pair
                try:
                    origin_data = o_data[origin_id]
                except KeyError:
                    # This should never happen because we should have preprocessed this out.
                    self.logger.debug(
                        f"Origin from OD Pairs not found in inputs. Skipped pair {od_pair}.")
                    continue
                route_name = f"{origin_id} - {dest_id}"
                icur.insertRow((route_name, 1, origin_id) + origin_data)

        # Insert destinations from each OD pair into the Route analysis
        with self.rt_solver.insertCursor(
            arcpy.nax.RouteInputDataType.Stops,
            ["RouteName", "Sequence", self.dest_unique_id_field_name, "SHAPE@"] + self.destination_transfer_fields
        ) as icur:
            for od_pair in self.od_pairs:
                origin_id, dest_id = od_pair
                try:
                    dest_data = d_data[dest_id]
                except KeyError:
                    # This should never happen because we should have preprocessed this out.
                    self.logger.debug(
                        f"Destination from OD Pairs not found in inputs. Skipped pair {od_pair}.")
                    continue
                route_name = f"{origin_id} - {dest_id}"
                icur.insertRow((route_name, 2, dest_id) + dest_data)

    def solve(self, chunk_definition):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        """Create and solve a Route analysis for the designated preassigned origin-destination pairs.

        Args:
            chunk_definition (list): For one-to-one, the ObjectID range to select from the input origins. For
                many-to-many, a list of [chunk starting row number, chunk size].
        """
        # Select the inputs to process
        if self.pair_type is helpers.PreassignedODPairType.one_to_one:
            self._select_inputs_one_to_one(chunk_definition)
        elif self.pair_type is helpers.PreassignedODPairType.many_to_many:
            self._get_od_pairs_for_chunk(chunk_definition)
            self._select_inputs_many_to_many()
        else:
            raise NotImplementedError(f"Invalid PreassignedODPairType: {self.pair_type}")

        # Initialize the Route solver object
        self.initialize_rt_solver()
        self._add_unique_id_fields()

        # Insert the origins and destinations
        self.logger.debug(f"Route solver fields transferred from Origins: {self.origin_transfer_fields}")
        self.logger.debug(f"Route solver fields transferred from Destinations: {self.destination_transfer_fields}")
        if self.pair_type is helpers.PreassignedODPairType.one_to_one:
            self._insert_stops_one_to_one()
        elif self.pair_type is helpers.PreassignedODPairType.many_to_many:
            self._insert_stops_many_to_many()
        else:
            raise NotImplementedError(f"Invalid PreassignedODPairType: {self.pair_type}")

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
                class_type = arcpy.nax.RouteInputDataType.PolygonBarriers
            elif shape_type == "Polyline":
                class_type = arcpy.nax.RouteInputDataType.LineBarriers
            elif shape_type == "Point":
                class_type = arcpy.nax.RouteInputDataType.PointBarriers
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
        self._export_to_feature_class(chunk_definition)

        self.logger.debug("Finished calculating Route.")

    def _export_to_feature_class(self, chunk_definition):
        """Export the Route result to a feature class."""
        # Make output gdb
        self.logger.debug("Creating output geodatabase for Route results...")
        od_workspace = os.path.join(self.job_folder, "scratch.gdb")
        helpers.run_gp_tool(
            self.logger,
            arcpy.management.CreateFileGDB,
            [os.path.dirname(od_workspace), os.path.basename(od_workspace)],
        )

        # Export routes
        output_routes = os.path.join(od_workspace, f"Routes_{chunk_definition[0]}_{chunk_definition[1]}")
        self.logger.debug(f"Exporting Route Routes output to {output_routes}...")
        self.solve_result.export(arcpy.nax.RouteOutputDataType.Routes, output_routes)

        # Export stops
        output_stops = os.path.join(od_workspace, f"Stops_{chunk_definition[0]}_{chunk_definition[1]}")
        self.logger.debug(f"Exporting Route Stops output to {output_stops}...")
        self.solve_result.export(arcpy.nax.RouteOutputDataType.Stops, output_stops)

        # Join the input ID fields to Routes
        # The new FirstStopID and LastStopID fields were added at Pro 3.1 / Enterprise 11.1 to make relationships
        # between IDs/OIDs in output classes are more reliable.  Use these fields if they exist in the output.
        # Otherwise, use FirstStopOID and LastStopOID, which are mostly reliable but not perfect.  For best results, use
        # the most recent ArcGIS software.
        if "FirstStopID" in self.solve_result.fieldNames(arcpy.nax.RouteOutputDataType.Routes):
            id_field_prefix = "ID"
        else:
            id_field_prefix = "OID"
        if self.reverse_direction:
            first_stop_field = self.dest_unique_id_field_name
            second_stop_field = self.origin_unique_id_field_name
        else:
            first_stop_field = self.origin_unique_id_field_name
            second_stop_field = self.dest_unique_id_field_name
        helpers.run_gp_tool(
            self.logger,
            arcpy.management.JoinField,
            [output_routes, f"FirstStop{id_field_prefix}", output_stops, "ObjectID", [first_stop_field]]
        )
        helpers.run_gp_tool(
            self.logger,
            arcpy.management.JoinField,
            [output_routes, f"LastStop{id_field_prefix}", output_stops, "ObjectID", [second_stop_field]]
        )

        self.job_result["outputRoutes"] = output_routes

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

    def teardown_logger(self):
        """Clean up and close the logger."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def solve_route(inputs, chunk):
    """Solve a Route analysis for the given inputs for the given chunk of preassigned OD pairs.

    Args:
        inputs (dict): Dictionary of keyword inputs suitable for initializing the Route class
        chunk (list): For one-to-one, the ObjectID range to select from the input origins. For many-to-many, a list of
            [chunk starting row number, chunk size].

    Returns:
        dict: Dictionary of results from the Route class
    """
    rt = Route(**inputs)
    if inputs["pair_type"] is helpers.PreassignedODPairType.one_to_one:
        rt.logger.info(f"Processing origins OID {chunk[0]} to {chunk[1]} as job id {rt.job_id}")
    elif inputs["pair_type"] is helpers.PreassignedODPairType.many_to_many:
        rt.logger.info(f"Processing chunk {chunk[0]} as job id {rt.job_id}")
    rt.solve(chunk)
    rt.teardown_logger()
    return rt.job_result


class ParallelRoutePairCalculator:  # pylint:disable = too-many-instance-attributes, too-few-public-methods
    """Solves a large Route by chunking the problem, solving in parallel, and combining results."""

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, pair_type_str, origins, origin_id_field, destinations, dest_id_field,
        network_data_source, travel_mode, time_units, distance_units,
        max_routes, max_processes, out_routes, scratch_folder, reverse_direction=False,
        assigned_dest_field=None, od_pair_table=None, time_of_day=None, barriers=None
    ):
        """Compute Routes between origins and their assigned destinations in parallel and combine results.

        Compute Routes in parallel and combine the results.
        This class assumes that the inputs have already been pre-processed and validated.

        Args:
            pair_type_str (str): String representation of the preassigned origin-destination pair type. Must match one
                of the enum names in helpers.PreassignedODPairType.
            origins (str, layer): Catalog path or layer for the input origins
            origin_id_field (str): Unique ID field of the input origins
            destinations (str, layer): Catalog path or layer for the input destinations
            dest_id_field: (str): Unique ID field of the input destinations
            network_data_source (str, layer): Catalog path, layer, or URL for the input network dataset
            travel_mode (str, travel mode): Travel mode object, name, or json string representation
            time_units (str): String representation of time units
            distance_units (str): String representation of distance units
            max_routes (int): Maximum number of origin-destination pairs that can be in one chunk
            max_processes (int): Maximum number of allowed parallel processes
            out_routes (str): Catalog path to the output routes feature class
            scratch_folder (str): Catalog path to the folder where intermediate outputs will be written.
            reverse_direction (bool, optional): Whether to reverse the direction of travel and calculate routes from
                destination to origin instead of origin to destination. Defaults to False. Only applicable for the
                one_to_one pair type.
            assigned_dest_field (str): Field in the input origins with the assigned destination ID
            od_pair_table (str): CSV file containing preassigned origin-destination pairs. Must have no headers. The
                first column contains origin ID, and the second column contains destination IDs.
            time_of_day (str): String representation of the start time for the analysis (required format defined in
                helpers.DATETIME_FORMAT)
            barriers (list(str, layer), optional): List of catalog paths or layers for point, line, and polygon barriers
                to use. Defaults to None.
        """
        pair_type = helpers.PreassignedODPairType[pair_type_str]
        self.origins = origins
        self.destinations = destinations
        self.out_routes = out_routes
        self.scratch_folder = scratch_folder
        time_units = helpers.convert_time_units_str_to_enum(time_units)
        distance_units = helpers.convert_distance_units_str_to_enum(distance_units)
        if not barriers:
            barriers = []
        self.max_processes = max_processes
        if not time_of_day:
            time_of_day = None
        else:
            time_of_day = datetime.datetime.strptime(time_of_day, helpers.DATETIME_FORMAT)

        # Initialize the dictionary of inputs to send to each OD solve
        self.rt_inputs = {
            "pair_type": pair_type,
            "origins": self.origins,
            "origin_id_field": origin_id_field,
            "destinations": self.destinations,
            "dest_id_field": dest_id_field,
            "network_data_source": network_data_source,
            "travel_mode": travel_mode,
            "time_units": time_units,
            "distance_units": distance_units,
            "time_of_day": time_of_day,
            "reverse_direction": reverse_direction,
            "scratch_folder": self.scratch_folder,
            "assigned_dest_field": assigned_dest_field,
            "od_pair_table": od_pair_table,
            "barriers": barriers,
            "origin_transfer_fields": [],  # Populate later
            "destination_transfer_fields": []  # Populate later
        }

        # List of intermediate output OD Line files created by each process
        self.route_fcs = []

        # Construct OID ranges for chunks of origins and destinations
        if pair_type is helpers.PreassignedODPairType.one_to_one:
            # Chunks are of the format [first origin ID, second origin ID]
            self.chunks = helpers.get_oid_ranges_for_input(origins, max_routes)
        elif pair_type is helpers.PreassignedODPairType.many_to_many:
            # Chunks are of the format [chunk_num, chunk_size]
            num_od_pairs = 0
            with open(od_pair_table, "r", encoding="utf-8") as f:
                for _ in f:
                    num_od_pairs += 1
            num_chunks = ceil(num_od_pairs / max_routes)
            self.chunks = [[i, max_routes] for i in range(num_chunks)]

        # Calculate the total number of jobs to use in logging
        self.total_jobs = len(self.chunks)

        self.optimized_cost_field = None

    def _validate_route_settings(self):
        """Validate Route settings before spinning up a bunch of parallel processes doomed to failure.

        Also check which field name in the output OD Lines will store the optimized cost values. This depends on the
        travel mode being used by the analysis, and we capture it here to use in later steps.
        """
        # Create a dummy Route object and set properties. This allows us to
        # detect any errors prior to spinning up a bunch of parallel processes and having them all fail.
        LOGGER.debug("Validating Route settings...")
        rt = None
        try:
            rt = Route(**self.rt_inputs)
            rt.initialize_rt_solver()
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
                rt.teardown_logger()
                # Delete output folder
                shutil.rmtree(rt.job_result["jobFolder"], ignore_errors=True)
                del rt

    def _populate_input_data_transfer_fields(self):
        """Discover if the origins and destinations include valid fields we can use in the Route analysis.

        Any fields with the correct names and data types matching valid fields recognized by the Route solver for the
        Stops input can be used in the analysis.  Compare the input origins and destinations fields with the list of
        supported Route Stops fields and populate the list of fields to transfer in the route inputs dictionary.
        """
        # Valid fields for the Route Stops input are described here:
        # https://pro.arcgis.com/en/pro-app/latest/arcpy/network-analyst/route-input-data-types.htm
        # Do not transfer RouteName or Sequence as these are explicitly controlled by this tool.  Do not transfer
        # LocationType because we want all inputs to be Stops. Waypoints don't make sense for this analysis.
        int_types = ["Integer", "SmallInteger"]
        numerical_types = ["Double", "Single"] + int_types
        rt_stops_input_fields = {
            "Name": ["String"],
            "AdditionalTime": numerical_types,
            "AdditionalDistance": numerical_types,
            "AdditionalCost": numerical_types,
            "TimeWindowStart": ["Date"],
            "TimeWindowEnd": ["Date"],
            "CurbApproach": int_types,
            "Bearing": numerical_types,
            "BearingTol": numerical_types,
            "NavLatency": numerical_types,
            "SourceID": int_types,
            "SourceOID": int_types,
            "PosAlong": numerical_types,
            "SideOfEdge": int_types
        }
        # Preserve origin and destination input fields that match names and types
        origin_transfer_fields = [
            f.name for f in arcpy.ListFields(self.origins) if f.name in rt_stops_input_fields and
            f.type in rt_stops_input_fields[f.name]]
        self.rt_inputs["origin_transfer_fields"] = origin_transfer_fields
        if origin_transfer_fields:
            LOGGER.info((
                "Supported fields in the input Origins table that will be used in the analysis: "
                f"{origin_transfer_fields}"
            ))
        destination_transfer_fields = [
            f.name for f in arcpy.ListFields(self.destinations) if f.name in rt_stops_input_fields and
            f.type in rt_stops_input_fields[f.name]]
        self.rt_inputs["destination_transfer_fields"] = destination_transfer_fields
        if destination_transfer_fields:
            LOGGER.info((
                "Supported fields in the input Destinations table that will be used in the analysis: "
                f"{destination_transfer_fields}"
            ))

    def solve_route_in_parallel(self):
        """Solve the Route in chunks and post-process the results."""
        # Validate Route settings. Essentially, create a dummy Route class instance and set up the
        # solver object to ensure this at least works. Do this up front before spinning up a bunch of parallel processes
        # that are guaranteed to all fail.
        self._validate_route_settings()

        # Check if the input origins and destinations have any fields we should use in the route analysis
        self._populate_input_data_transfer_fields()

        # Compute Route in parallel
        LOGGER.info(f"Beginning parallelized Route solves ({self.total_jobs} chunks)")
        completed_jobs = 0  # Track the number of jobs completed so far to use in logging
        # Use the concurrent.futures ProcessPoolExecutor to spin up parallel processes that solve the routes
        with futures.ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            # Each parallel process calls the solve_route() function with the rt_inputs dictionary for the
            # given origin ranges and their assigned destinations.
            jobs = {executor.submit(solve_route, self.rt_inputs, range): range for range in self.chunks}
            # As each job is completed, add some logging information and store the results to post-process later
            for future in futures.as_completed(jobs):
                try:
                    # The Route job returns a results dictionary. Retrieve it.
                    result = future.result()
                except Exception:  # pylint: disable=broad-except
                    # If we couldn't retrieve the result, some terrible error happened and the job errored.
                    # Note: This does not mean solve failed. It means some unexpected error was thrown. The most likely
                    # causes are:
                    # a) If you're calling a service, the service was temporarily down.
                    # b) You had a temporary file read/write or resource issue on your machine.
                    # c) If you're actively updating the code, you introduced an error.
                    # To make the tool more robust against temporary glitches, retry submitting the job up to the number
                    # of times designated in helpers.MAX_RETRIES.  If the job is still erroring after that many retries,
                    # fail the entire tool run.
                    errs = traceback.format_exc().splitlines()
                    failed_range = jobs[future]
                    LOGGER.debug((
                        f"Failed to get results for Route chunk {failed_range} from the parallel process. Will retry "
                        f"up to {helpers.MAX_RETRIES} times. Errors: {errs}"
                    ))
                    job_failed = True
                    num_retries = 0
                    while job_failed and num_retries < helpers.MAX_RETRIES:
                        num_retries += 1
                        try:
                            future = executor.submit(solve_route, self.rt_inputs, failed_range)
                            result = future.result()
                            job_failed = False
                            LOGGER.debug(f"Route chunk {failed_range} succeeded after {num_retries} retries.")
                        except Exception:  # pylint: disable=broad-except
                            # Update exception info to the latest error
                            errs = traceback.format_exc().splitlines()
                    if job_failed:
                        # The job errored and did not succeed after retries.  Fail the tool run because something
                        # terrible is happening.
                        LOGGER.debug(f"Route chunk {failed_range} continued to error after {num_retries} retries.")
                        LOGGER.error("Failed to get Route result from parallel processing.")
                        errs = traceback.format_exc().splitlines()
                        for err in errs:
                            LOGGER.error(err)
                        raise

                # If we got this far, the job completed successfully and we retrieved results.
                completed_jobs += 1
                LOGGER.info(
                    f"Finished Route calculation {completed_jobs} of {self.total_jobs}.")

                # Parse the results dictionary and store components for post-processing.
                if result["solveSucceeded"]:
                    self.route_fcs.append(result["outputRoutes"])
                else:
                    # Typically, a solve fails because no destinations were found for any of the origins in the chunk,
                    # and this is a perfectly legitimate failure. It is not an error. However, they may be other, less
                    # likely, reasons for solve failure. Write solve messages to the main GP message thread in debug
                    # mode only in case the user is having problems. The user can also check the individual OD log
                    # files.
                    LOGGER.debug(f"Solve failed for job id {result['jobId']}.")
                    LOGGER.debug(result["solveMessages"])

        # Post-process outputs
        if self.route_fcs:
            LOGGER.info("Post-processing Route results...")
            self.route_fcs = sorted(self.route_fcs)
            self._post_process_route_fcs()
        else:
            LOGGER.warning("All Route solves failed, so no output was produced.")

        # Clean up
        # Delete the job folders if the job succeeded
        if DELETE_INTERMEDIATE_OUTPUTS:
            LOGGER.info("Deleting intermediate outputs...")
            try:
                shutil.rmtree(self.scratch_folder, ignore_errors=True)
            except Exception:  # pylint: disable=broad-except
                # If deletion doesn't work, just throw a warning and move on. This does not need to kill the tool.
                LOGGER.warning(f"Unable to delete intermediate Route output folder {self.scratch_folder}.")

        LOGGER.info("Finished calculating Routes.")

    def _post_process_route_fcs(self):
        """Merge the routes calculated in each separate process into a single feature class.

        Create an empty final output feature class and populate it using InsertCursor, as this tends to be faster than
        using the Merge geoprocessing tool.
        """
        # Create the final output feature class
        desc = arcpy.Describe(self.route_fcs[0])
        helpers.run_gp_tool(
            LOGGER,
            arcpy.management.CreateFeatureclass, [
                os.path.dirname(self.out_routes),
                os.path.basename(self.out_routes),
                "POLYLINE",
                self.route_fcs[0],  # template feature class to transfer full schema
                "SAME_AS_TEMPLATE",
                "SAME_AS_TEMPLATE",
                desc.spatialReference
            ]
        )

        # Insert the rows from all the individual output feature classes into the final output
        fields = ["SHAPE@"] + [f.name for f in desc.fields]
        with arcpy.da.InsertCursor(self.out_routes, fields) as cur:  # pylint: disable=no-member
            for fc in self.route_fcs:
                for row in arcpy.da.SearchCursor(fc, fields):  # pylint: disable=no-member
                    cur.insertRow(row)


def launch_parallel_rt_pairs():
    """Read arguments passed in via subprocess and run the parallel Route.

    This script is intended to be called via subprocess via the solve_large_route_pair_analysis.py module, which does
    essential preprocessing and validation. Users should not call this script directly from the command line.

    We must launch this script via subprocess in order to support parallel processing from an ArcGIS Pro script tool,
    which cannot do parallel processing directly.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description=globals().get("__doc__", ""), fromfile_prefix_chars='@')

    # Define Arguments supported by the command line utility

    # --pair-type parameter
    help_string = "The type of origin-destination pair assignment to use. Either one_to_one or many_to_many."
    parser.add_argument("-pt", "--pair-type", action="store", dest="pair_type_str", help=help_string, required=True)

    # --origins parameter
    help_string = "The full catalog path to the feature class containing the origins."
    parser.add_argument("-o", "--origins", action="store", dest="origins", help=help_string, required=True)

    # --origins-id-field parameter
    help_string = "The name of the unique ID field in origins."
    parser.add_argument(
        "-oif", "--origins-id-field", action="store", dest="origin_id_field", help=help_string, required=True)

    # --destinations parameter
    help_string = "The full catalog path to the feature class containing the destinations."
    parser.add_argument("-d", "--destinations", action="store", dest="destinations", help=help_string, required=True)

    # --destinations-id-field parameter
    help_string = "The name of the unique ID field in destinations."
    parser.add_argument(
        "-dif", "--destinations-id-field", action="store", dest="dest_id_field", help=help_string, required=True)

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

    # --max-routes parameter
    help_string = "Maximum number of routes that can be in one chunk for parallel processing of Route solves."
    parser.add_argument(
        "-mr", "--max-routes", action="store", dest="max_routes", type=int, help=help_string, required=True)

    # --max-processes parameter
    help_string = "Maximum number parallel processes to use for the Route solves."
    parser.add_argument(
        "-mp", "--max-processes", action="store", dest="max_processes", type=int, help=help_string, required=True)

    # --reverse-direction parameter
    help_string = "Whether to reverse the direction of travel (destination to origin)."
    parser.add_argument(
        "-rd", "--reverse-direction", action="store", type=lambda x: bool(strtobool(x)),
        dest="reverse_direction", help=help_string, required=True)

    # --out-routes parameter
    help_string = "The full catalog path to the output routes feature class."
    parser.add_argument("-r", "--out-routes", action="store", dest="out_routes", help=help_string, required=True)

    # --scratch-folder parameter
    help_string = "The full catalog path to the scratch folder where intermediate outputs will be stored."
    parser.add_argument(
        "-sf", "--scratch-folder", action="store", dest="scratch_folder", help=help_string, required=True)

    # --assigned-dest-field parameter
    help_string = ("The name of the field in origins indicating the assigned destination. "
                   "Required for one_to_one pair-type")
    parser.add_argument(
        "-adf", "--assigned-dest-field", action="store", dest="assigned_dest_field", help=help_string, required=False)

    # --od-pair-table parameter
    help_string = "CSV file holding preassigned OD pairs. Required for many_to_many pair-type."
    parser.add_argument(
        "-odp", "--od-pair-table", action="store", dest="od_pair_table", help=help_string, required=False)

    # --time-of-day parameter
    help_string = (f"The time of day for the analysis. Must be in {helpers.DATETIME_FORMAT} format. Set to None for "
                   "time neutral.")
    parser.add_argument("-tod", "--time-of-day", action="store", dest="time_of_day", help=help_string, required=False)

    # --barriers parameter
    help_string = "A list of catalog paths to the feature classes containing barriers to use in the Route."
    parser.add_argument(
        "-b", "--barriers", action="store", dest="barriers", help=help_string, nargs='*', required=False)

    try:
        # Get arguments as dictionary.
        args = vars(parser.parse_args())

        # Initialize a parallel Route calculator class
        rt_calculator = ParallelRoutePairCalculator(**args)
        # Solve the Route in parallel chunks
        start_time = time.time()
        rt_calculator.solve_route_in_parallel()
        LOGGER.info(f"Parallel Route calculation completed in {round((time.time() - start_time) / 60, 2)} minutes")

    except Exception:  # pylint: disable=broad-except
        LOGGER.error("Error in parallelization subprocess.")
        errs = traceback.format_exc().splitlines()
        for err in errs:
            LOGGER.error(err)
        raise


if __name__ == "__main__":
    # This script should always be launched via subprocess as if it were being called from the command line.
    launch_parallel_rt_pairs()
