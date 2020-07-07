"""Compute Origin Destination (OD) cost matrix and save the output matrix as a feature class."""
################################################################################
'''Copyright 2020 Esri
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'''
################################################################################

from concurrent import futures
import os
import uuid
import logging
import shutil
import itertools
import time
import argparse
from collections import namedtuple

import arcpy
import arcgisscripting

# Module level logger
logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
logger.addHandler(logging.NullHandler())

# These fields will be added to the input origin and destination feature classes and will be calculated to equal
# the original OID so we can later reliably join between the final OD Lines output and these original inputs.
ORIGINS_UNIQUE_ID_FIELD = "OriginalOriginsOID"
DESTINATIONS_UNIQUE_ID_FIELD = "OriginalDestinationsOID"


class ODCostMatrix:  # pylint:disable = too-many-instance-attributes
    """Solve a OD Cost Matrix problem."""

    def __init__(self, **kwargs):
        """Set up names used in other methods."""
        # Store keyword args as instance attributes
        self.origins = kwargs["origins"]
        self.destinations = kwargs["destinations"]
        self.network_data_source = kwargs["network_data_source"]
        self.travel_mode = kwargs["travel_mode"]
        self.output_folder = kwargs["output_folder"]
        self.destination_count = kwargs.get("destination_count", None)
        self.cutoff = kwargs.get("cutoff", None)
        self.target_count = kwargs.get("target_count", None)

        # Setup the class logger
        cls_logger = logging.getLogger(__name__)
        self.setup_logger(cls_logger)
        self.logger = cls_logger

        # other instance attributes
        self.portal_desc = kwargs.get("portal_description", {})
        self.job_id = uuid.uuid4().hex
        self.job_folder = os.path.join(self.output_folder, self.job_id)
        self.pid = os.getpid()
        os.mkdir(self.job_folder)
        self.origins_sublayer = None
        self.destinations_sublayer = None
        self.lines_sublayer = None
        self.is_service = self.is_nds_service(self.network_data_source)
        self.time_attribute = ""
        self.distance_attribute = ""
        self.is_travel_mode_time_based = True
        self.input_origins_layer = "InputOrigins" + self.job_id
        self.input_destinations_layer = "InputDestinations" + self.job_id
        self.input_origins_layer_obj = None
        self.input_destinations_layer_obj = None
        self.job_result = {  # Store information about each OD cost matrix result
            "jobId": self.job_id,
            "jobFolder": self.job_folder,
            "solveSucceeded": False,
            "solveMessages": "",
            "outputLines": "",
            "outputLayerFile": ""
        }
        # Create a unique workspace that will contains inputs and outputs for OD cost matrix computation
        result = arcpy.management.CreateFileGDB(self.job_folder, "scratch")
        self.od_workspace = result.getOutput(0)

        # Create a network dataset layer
        self.nds_layer_name = "NetworkDatasetLayer"
        self._make_nds_layer()

        # Get the ObjectID fields for origins and destinations
        desc_origins = arcpy.Describe(self.origins)
        desc_destinations = arcpy.Describe(self.destinations)
        self.origins_oid_field_name = desc_origins.oidFieldName
        self.destinations_oid_field_name = desc_destinations.oidFieldName
        # These candidate fields are used for loading origins and desinations. They allow us to pass through the unique
        # ID fields that we're adding to the outputs so we can join the output Lines back to the original input data.
        self.origins_candidate_fields = [f for f in desc_origins.fields if f.name == ORIGINS_UNIQUE_ID_FIELD]
        self.destinations_candidate_fields = [
            f for f in desc_destinations.fields if f.name == DESTINATIONS_UNIQUE_ID_FIELD]

        # Get the impedance, time impedance and distance impedance from the travel mode
        self._get_travel_mode_info()

    def solve(self, origins_criteria,  # pylint:disable = too-many-locals, too-many-statements
              destinations_criteria):
        """Generate a origin destination cost matrix using a network data source.

        Args:
            network_data_source: The network dataset layer or the portal URL for the network dataset source used to
                                 compute the origin destination cost matrix
            origins_criteria: A two value tuple representing the range of object ids for the origins to process.
                              For example, to process origins with object id between 101 and 200, pass (101, 200)
            destinations_criteria: A two value tuple representing the range of object ids for the destinations to
                              process. For example, to process destinations with object id between 101 and 200,
                              pass (101, 200)

        """
        # Set the workspace that will contains input and output NA classes used by OD Cost Matrix solver
        arcpy.env.workspace = self.od_workspace

        # Determine if we need to use the network dataset layer pointing to a local network dataset or a portal url as
        # our network data source. For a local network dataset, we also need to checkout the network analyst extension
        # license.
        if self.is_service:
            network_data_source = self.network_data_source
        else:
            network_data_source = self.nds_layer_name
            arcpy.CheckOutExtension("network")

        # Create a new OD cost matrix object
        self.logger.debug("Creating OD Cost Matrix object")
        od_solver = arcpy.nax.OriginDestinationCostMatrix(network_data_source)

        # Set the OD cost matrix analysis properties. Some of these are set based on user inputs. Others are
        # hard-coded here.  You should modify these to suit your analysis needs. Read about the OD Cost Matrix object's
        # properties and settings in the documentation:
        # https://pro.arcgis.com/en/pro-app/arcpy/network-analyst/odcostmatrix.htm
        od_solver.allowSaveLayerFile = True
        od_solver.travelMode = self.travel_mode
        od_solver.defaultDestinationCount = self.target_count
        od_solver.distanceUnits = arcpy.nax.DistanceUnits.Miles
        od_solver.timeUnits = arcpy.nax.TimeUnits.Minutes
        od_solver.defaultImpedanceCutoff = self.cutoff
        od_solver.accumulateAttributeNames = [self.time_attribute, self.distance_attribute]
        od_solver.lineShapeType = arcpy.nax.LineShapeType.NoLine
        od_solver.searchTolerance = 20000
        od_solver.searchToleranceUnits = arcpy.nax.DistanceUnits.Meters

        # Select the origins and destinations to process
        self._select_inputs(origins_criteria, destinations_criteria)

        # Map the unique ID field from input origin to the Name field on the origins sub layer and the unique ID field
        # from input destinations to the Name field on the destinations sub layer. Map the network locations fields
        # using field mappings. Use the candidate field mappings with the unique ID field to pass through those unique
        # IDs to the input and then to the output.
        origins_field_mappings = od_solver.fieldMappings(arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins,
                                                         True, self.origins_candidate_fields)
        origins_field_mappings["Name"].mappedFieldName = ORIGINS_UNIQUE_ID_FIELD
        destinations_field_mappings = od_solver.fieldMappings(
            arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
            True, self.destinations_candidate_fields)
        destinations_field_mappings["Name"].mappedFieldName = DESTINATIONS_UNIQUE_ID_FIELD

        # Load the origins and destinations using the field mappings and a search tolerance of 20000 Meters.
        self.logger.debug("Loading origins and destinations")
        od_solver.load(arcpy.nax.OriginDestinationCostMatrixInputDataType.Origins, self.input_origins_layer_obj,
                       origins_field_mappings, False)
        od_solver.load(arcpy.nax.OriginDestinationCostMatrixInputDataType.Destinations,
                       self.input_destinations_layer_obj, destinations_field_mappings, False)

        # Solve the OD cost matrix analysis
        self.logger.debug("Solving OD cost matrix")
        solve_start = time.time()
        solve_result = od_solver.solve()
        solve_end = time.time()
        solve_msgs = "\n".join([msg[-1] for msg in solve_result.solverMessages(arcpy.nax.MessageSeverity.All)])
        self.job_result["solveMessages"] = solve_msgs
        if not solve_result.solveSucceeded:
            return
        self.job_result["solveSucceeded"] = True
        self.logger.debug("Solving OD cost matrix completed in %s (seconds)", round(solve_end - solve_start, 3))
        lyr_file = os.path.join(self.job_folder, "result_{}.lyr".format(self.job_id))
        self.job_result["outputLayerFile"] = lyr_file
        solve_result.saveAsLayerFile(lyr_file)

        # Export the ODlines and output origins and destinations to feature classes
        output_od_origins = os.path.join(self.od_workspace, "output_od_origins")
        solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Origins, output_od_origins)
        output_od_destinations = os.path.join(self.od_workspace, "output_od_destinations")
        solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Destinations, output_od_destinations)
        output_od_lines = os.path.join(self.od_workspace, "output_od_lines")
        solve_result.export(arcpy.nax.OriginDestinationCostMatrixOutputDataType.Lines, output_od_lines)
        # Join unique ID fields to Lines so the original origin and destination OIDs are preserved in the output lines
        arcpy.management.JoinField(
            output_od_lines, "OriginOID", output_od_origins, "OriginOID", [ORIGINS_UNIQUE_ID_FIELD])
        arcpy.management.JoinField(
            output_od_lines, "DestinationOID", output_od_destinations, "DestinationOID", [DESTINATIONS_UNIQUE_ID_FIELD])
        self.job_result["outputLines"] = output_od_lines

    def _select_inputs(self, origins_criteria, destinations_criteria):
        """Select origins and destinations to process."""
        # Select the origins and destinations to process
        origins_where_clause = "{} >= {} And {} <= {}".format(self.origins_oid_field_name, origins_criteria[0],
                                                              self.origins_oid_field_name, origins_criteria[1])
        self.input_origins_layer_obj = arcpy.management.MakeFeatureLayer(
            self.origins, self.input_origins_layer, origins_where_clause).getOutput(0)
        if self.cutoff:
            # select destinations within the cutoff from origins
            arcpy.management.MakeFeatureLayer(self.destinations, self.input_destinations_layer)
            # Convert from travel time to distance if needed.  Assume a very fast travel speed of 100 miles per hour to
            # give a safe buffer distance and make sure all relevant points get included. Note that we are assuming
            # the cutoff is specified in units of either Minutes or Miles. If you want to use different units, you need
            # to modify the code here and also change the distanceUnits and timeUnits properties of the OD Cost Matrix
            # object.
            minutes_to_miles = 1.6667  # Travel speed of 100 miles per hour.
            cutoff = self.cutoff * minutes_to_miles if self.is_travel_mode_time_based else self.cutoff
            result = arcpy.management.SelectLayerByLocation(self.input_destinations_layer, "WITHIN_A_DISTANCE_GEODESIC",
                                                            self.input_origins_layer, "{} Miles".format(cutoff),)
            # If no destinations are within the cutoff, skip this iteration
            if not result.getOutput(0).getSelectionSet():
                msg = "No destinations found within the cutoff"
                self.logger.warning(msg)
                self.job_result["solveMessages"] = msg
                return
        else:

            destinations_where_clause = "{} >= {} And {} <= {}".format(self.destinations_oid_field_name,
                                                                       destinations_criteria[0],
                                                                       self.destinations_oid_field_name,
                                                                       destinations_criteria[1])
            self.input_destinations_layer_obj = arcpy.management.MakeFeatureLayer(
                self.destinations, self.input_destinations_layer, destinations_where_clause).getOutput(0)

    def _get_travel_mode_info(self):
        """Get additional info from the travel mode."""
        # When working with services, get the travel modes defined in the portal
        if self.is_service:
            travel_modes = arcpy.na.GetTravelModes(self.network_data_source)
        else:
            travel_modes = arcpy.na.GetTravelModes(self.nds_layer_name)
        travel_mode = travel_modes[self.travel_mode]
        impedance = travel_mode.impedance
        self.time_attribute = travel_mode.timeAttributeName
        self.distance_attribute = travel_mode.distanceAttributeName
        self.is_travel_mode_time_based = True if self.time_attribute == impedance else False

    def _make_nds_layer(self):
        """Create a network dataset layer if one does not exist."""
        # Can only create a layer for a local network dataset
        if self.is_service:
            return
        if arcpy.Exists(self.nds_layer_name):
            self.logger.debug("Using existing network dataset layer: %s", self.nds_layer_name)
        else:
            self.logger.debug("Creating network dataset layer")
            arcpy.na.MakeNetworkDatasetLayer(self.network_data_source, self.nds_layer_name)

    @staticmethod
    def get_nds_search_criteria(network_data_source):  # pylint:disable = too-many-locals
        """Return the search criteria for a network dataset that can be used with Calculate Locations GP tool.

        Note: This method is not needed in ArcGIS Pro 2.6 or later because the Calculate Locations tool was given a
        good default value for this parameter at that release.

        Args:
            network_data_source: The catalog path to the network dataset
        Returns:
            The search criteria for the netrwork dataset.

        """
        # Determine if a network source defines subtypes. The search criteria needs to be specified for each subtype
        # using the following pattern ["SourceName : subtype description", "SHAPE"]
        tmp_search_criteria = []
        search_criteria = []
        nds_desc = arcpy.Describe(network_data_source)
        nds_fds_path = os.path.dirname(nds_desc.catalogPath)
        for src in nds_desc.sources:
            src_element_type = src.elementType
            if src_element_type not in ("Edge", "Junction"):
                continue
            src_name = src.name
            src_path = os.path.join(nds_fds_path, src_name)
            src_desc = arcpy.Describe(src_path)
            if src_desc.subTypeFieldName:
                # Get the description of subtype codes
                src_sub_types = arcpy.da.ListSubtypes(src_path)  # pylint:disable = no-member
                for code in src_sub_types:
                    sub_type_description = src_sub_types[code]["Name"]
                    tmp_search_criteria.append([u"{} : {}".format(src_name, sub_type_description), src_element_type])
            else:
                tmp_search_criteria.append([src_name, src_element_type])
        # Set shape type for all edge sources to be SHAPE and NONE for all junction sources
        for criteria in tmp_search_criteria:
            shape_type = "SHAPE" if criteria[-1] == "Edge" else "NONE"
            search_criteria.append([criteria[0], shape_type])
        return search_criteria

    @staticmethod
    def preprocess_inputs(input_features, network_data_source, travel_mode, output_workspace, unique_id_field_name):
        """Preprocess input features so that they can be processed in chunks.

        The function performs tasks such as sptially sorting input features and calculate network locations for the
        features.

        Args:
            input_features: The full catalog path to the input feature class.
            network_data_source: The catalog path to the network dataset used for analysis
            travel_mode: Name of the travel mode used for the analysis.
            output_workspace: The catalog path of the output workspace in which to write the output feature class.
            unique_id_field_name: Field name of a unique ID to add and populate.
        Returns:
            The full catalog path of the processed feature class.

        """
        logger.info("Preprocessing %s", input_features)

        # Create output features in a feature class with the same name as input feature class.
        desc_input_features = arcpy.Describe(input_features)
        input_path = desc_input_features.catalogPath
        output_features = arcpy.CreateUniqueName(os.path.basename(input_path), output_workspace)

        # Add a unique ID field so we don't lose OID info when we sort
        logger.debug("Adding unique ID field for %s", input_features)
        if unique_id_field_name not in [f.name for f in desc_input_features.fields]:
            arcpy.management.AddField(input_features, unique_id_field_name, "LONG")
        arcpy.management.CalculateField(input_features, unique_id_field_name, f"!{desc_input_features.oidFieldName}!")

        # Spatially sort input features
        try:
            logger.debug("Spatially sorting %s", input_features)
            result = arcpy.management.Sort(input_features, output_features,
                                           [[desc_input_features.shapeFieldName, "ASCENDING"]], "PEANO")
            logger.debug(result.getMessages().split("\n")[-1])
        except arcgisscripting.ExecuteError:  # pylint:disable = no-member
            msgs = arcpy.GetMessages(2)
            if "000824" in msgs:  # ERROR 000824: The tool is not licensed.
                logger.debug("Skipping spatial sorting because the Advanced license is not available.")
            else:
                logger.debug("Skipping spatial sorting because the tool failed. Messages:\n%s", msgs)
            arcpy.management.Copy(input_features, output_features)

        # Calculate network location fields if network data source is local
        if not ODCostMatrix.is_nds_service(network_data_source):
            logger.debug("Calculating network locations for %s", input_features)
            result = arcpy.na.CalculateLocations(output_features, network_data_source, "20 Miles",
                                                 ODCostMatrix.get_nds_search_criteria(network_data_source),
                                                 travel_mode=travel_mode)
            logger.debug(result.getMessages().split("\n")[-1])

        return output_features

    @staticmethod
    def get_oid_ranges(origins_count, destinations_count, max_od_size):
        """Return an iterable of OIDs ranges for origins and destinations.

        Args:
            origins_count: Total number of origins.
            destinations_count: Total number of destinations.
            max_od_size: The maximum number of od lines that should be present in a single od cost matrix. This number
                        governs how many origins and destinations are processed in one iteration. For example, if you
                        have 500 origins and 100 destinations and maximum od size is 10,000, then each iteration will
                        process 100 origins and 100 destinations.
        Returns:
            An iterable where each element is a tuple containing lower and upper bound OIDs for origins and
            destinations to process in each iteration. For example get_oid_ranges(200, 100, 10000) will return an
            iterable such as [((1, 100), (1, 100)), ((101, 200), (1, 100))]

        """
        od_ranges = []
        if destinations_count <= max_od_size:
            max_origins = max_od_size // destinations_count
            origin_ranges = itertools.zip_longest(range(1, origins_count + 1, max_origins),
                                                  range(max_origins, origins_count + 1, max_origins),
                                                  fillvalue=origins_count)
            od_ranges = ((val, (1, destinations_count)) for val in origin_ranges)
        else:
            max_destinations = destinations_count - max_od_size
            max_origins = max_od_size // max_destinations
            origin_ranges = itertools.zip_longest(range(1, origins_count + 1, max_origins),
                                                  range(max_origins, origins_count + 1, max_origins),
                                                  fillvalue=origins_count)
            dest_ranges = itertools.zip_longest(range(1, destinations_count + 1, max_destinations),
                                                range(max_destinations, destinations_count + 1, max_destinations),
                                                fillvalue=destinations_count)
            od_ranges = itertools.product(origin_ranges, dest_ranges)
        return od_ranges

    @staticmethod
    def get_oid_ranges_agol(origins_count, destinations_count, max_origins, max_destinations):
        """Return an iterable of OIDs ranges for origins and destinations based on max origins and max destinations.

        Args:
            origins_count: Total number of origins.
            destinations_count: Total number of destinations.
            max_origins: The maximum number of origins supported by the agol od service.
            max_destinations: The maximum number of destinations supported by the agol od service.
        Returns:
            An iterable where each element is a tuple containing lower and upper bound OIDs for origins and
            destinations to process in each iteration. For example get_oid_ranges_agol(1500, 1200, 1000, 1000) will
            return an iterable such as [((1, 1000), (1, 1000)), ((1, 1000), (1001, 1200)),
                                        ((1001, 1500), (1, 1000)), ((1001, 1500), (1001, 1200))]

        """
        od_ranges = []
        if destinations_count <= max_destinations:
            origin_ranges = itertools.zip_longest(range(1, origins_count + 1, max_origins),
                                                  range(max_origins, origins_count + 1, max_origins),
                                                  fillvalue=origins_count)
            od_ranges = ((val, (1, destinations_count)) for val in origin_ranges)
        else:
            origin_ranges = itertools.zip_longest(range(1, origins_count + 1, max_origins),
                                                  range(max_origins, origins_count + 1, max_origins),
                                                  fillvalue=origins_count)
            dest_ranges = itertools.zip_longest(range(1, destinations_count + 1, max_destinations),
                                                range(max_destinations, destinations_count + 1, max_destinations),
                                                fillvalue=destinations_count)
            od_ranges = itertools.product(origin_ranges, dest_ranges)
        return od_ranges

    @staticmethod
    def setup_logger(logger_obj):
        """Set up the logger used for logging messages.

        Args:
            logger_obj: The logger instance.

        """
        logger_obj.setLevel(logging.DEBUG)
        # logger_obj.propagate = False
        if len(logger_obj.handlers) <= 1:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter("%(process)d | %(message)s")
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG)
            logger_obj.addHandler(console_handler)

    @staticmethod
    def is_nds_service(network_data_source):
        """Return true if the network data source points to a service."""
        return True if network_data_source.startswith("http") else False

    @staticmethod
    def get_tool_limits(portal_url, service_name="asyncODCostMatrix",
                        tool_name="GenerateOriginDestinationCostMatrix"):
        """Return a dictionary of various limits supported by a portal tool.

        Args:
            portal_url: The URL of the active portal that is used as the network dataset source.
            service_name: The name of the utility service configured with the portal to perform a given analysis.
            tool_name: The name of the tool within the service that performs the analysis.
        Returns:
            A dictionary with key as limit name and value as limit value.

        """
        if not portal_url.endswith("/"):
            portal_url = portal_url + "/"
        tool_limits = arcpy.na.GetWebToolInfo(service_name, tool_name, portal_url)
        return tool_limits["serviceLimits"]


def solve_od_cost_matrix(inputs, chunk):
    """Solve OD cost matrix on a seperate process for each iteration.

    Args:
        inputs: An iterable of dictinories containing the inputs such as origins and destinations to process for each
                iteration.
        chunk: An iterable of OID ranges for origins and destinations defining the origins and destinations that will be
               processed in each iteration.
    Returns:
        A dictionary which contains information about the result. The dictinory has the following keys:
            "jobId" -- A unique ID
            "jobFolder" -- Folder that stores intermidiate results
            "solveSucceeded" -- Status of the OD cost matrix solve
            "solveMessages" -- Messages from the OD cost matrix solve
            "outputLines" -- Catalog path to the feature class storing output OD cost matrix lines
            "outputLayerFile" -- Catalog path to the layer file for OD cost matrix solve

    """
    odcm = ODCostMatrix(**inputs)
    logger.info("Processing origins OID %s to %s and destinations OID %s to %s as job id %s",
                chunk[0][0], chunk[0][1], chunk[1][0], chunk[1][1], odcm.job_id)
    odcm.solve(chunk[0], chunk[1])
    result = odcm.job_result
    logger.debug("Saved OD lines at %s", result["outputLines"])
    return result


def main(**inputs):  # pylint:disable = too-many-locals, too-many-statements, too-many-branches
    """Preprocess inputs, compute OD cost matrix and postprocess outputs."""
    # Create the output workspace
    out_gdb_name = "outputs"
    out_gdb = os.path.join(inputs["output_folder"], out_gdb_name + ".gdb")
    if not os.path.exists(out_gdb):
        arcpy.management.CreateFileGDB(inputs["output_folder"], out_gdb_name)

    # Preprocess inputs
    pp_origins = ODCostMatrix.preprocess_inputs(inputs["origins"], inputs["network_data_source"], inputs["travel_mode"],
                                                out_gdb, ORIGINS_UNIQUE_ID_FIELD)
    pp_destinations = ODCostMatrix.preprocess_inputs(inputs["destinations"], inputs["network_data_source"],
                                                     inputs["travel_mode"], out_gdb, DESTINATIONS_UNIQUE_ID_FIELD)

    inputs["origins"] = pp_origins
    inputs["destinations"] = pp_destinations

    # Store count of input origins and destinations
    origins_count = int(arcpy.management.GetCount(inputs["origins"]).getOutput(0))
    destinations_count = int(arcpy.management.GetCount(inputs["destinations"]).getOutput(0))

    # Determine if working with online or enterprise portal
    network_data_source = inputs["network_data_source"]
    is_agol = False
    portal_desc = {}
    if ODCostMatrix.is_nds_service(network_data_source):
        logger.debug("Getting information from the portal")
        portal_desc = arcpy.GetPortalDescription(network_data_source)
        inputs["portal_description"] = portal_desc
        is_agol = not portal_desc["isPortal"]

    # Get iterables for the inputs
    if is_agol:
        # Get the max origins and max destinations if working with AGOL
        tool_limits = ODCostMatrix.get_tool_limits(network_data_source)
        max_origins = int(tool_limits["maximumOrigins"])
        max_destinations = int(tool_limits["maximumDestinations"])
        # Chunk origin and destination OID ranges based on max origins and max destinations
        ranges = ODCostMatrix.get_oid_ranges_agol(origins_count, destinations_count, max_origins, max_destinations)
        # Adjust properties specific to working with AGOL service.
        inputs["workers"] = min(4, inputs["workers"])
        inputs["max_od_size"] = max_origins * max_destinations
    else:
        ranges = ODCostMatrix.get_oid_ranges(origins_count, destinations_count, inputs["max_od_size"])

    inputs_iter = itertools.repeat(inputs)

    # Compute OD cost matrix
    od_line_fcs = []
    job_folders_to_delete = []
    # Run on multiple processes when solving large ODs
    if origins_count * destinations_count > inputs["max_od_size"]:
        with futures.ProcessPoolExecutor(max_workers=inputs["workers"]) as executors:
            results = executors.map(solve_od_cost_matrix, inputs_iter, ranges)
            # Compute a list of od results from each iteration if the solve is successful.
            for result in results:
                if result["solveSucceeded"]:
                    od_line_fcs.append(result["outputLines"])
                    job_folders_to_delete.append(result["jobFolder"])
                else:
                    logger.warning("Solve failed for job id %s", result["jobId"])
                    logger.debug(result["solveMessages"])
    else:
        result = solve_od_cost_matrix(inputs, [(1, origins_count), (1, destinations_count)])
        if result["solveSucceeded"]:
            od_line_fcs.append(result["outputLines"])
            job_folders_to_delete.append(result["jobFolder"])
        else:
            logger.warning("Solve failed for job id %s", result["jobId"])
            logger.debug(result["solveMessages"])

    # Merge individual OD matrix feature classes into a single feature class
    if od_line_fcs:
        output_fc = arcpy.CreateUniqueName("output_od_lines", out_gdb)
        logger.info("Merging results to %s", output_fc)
        result = arcpy.management.Merge(od_line_fcs, output_fc)
        logger.debug(result.getMessages().split("\n")[-1])

    # Cleanup
    # Delete the job folders if the job succeeded
    for folder in job_folders_to_delete:
        logger.debug("Deleting %s", folder)
        shutil.rmtree(folder, ignore_errors=True)

    # Delete the preprocessed inputs
    arcpy.management.Delete(pp_origins)
    arcpy.management.Delete(pp_destinations)


def _cli():
    """Command line interface for the tool."""
    # Create the parser
    parser = argparse.ArgumentParser(description=globals().get("__doc__", ""), fromfile_prefix_chars='@')

    # Define Arguments supported by the command line utility

    # --origins parameter
    help_string = "The full catalog path to the feature class contaning the origins."
    parser.add_argument("-o", "--origins", action="store", dest="origins", help=help_string, required=True)

    # --destinations parameter
    help_string = "The full catalog path to the feature class contaning the destinations."
    parser.add_argument("-d", "--destinations", action="store", dest="destinations", help=help_string, required=True)

    # --network-data-source parameter
    help_string = "The full catalog path to the network dataset or a portal url that will be used for the analysis."
    parser.add_argument("-n", "--network-data-source", action="store", dest="network_data_source", help=help_string,
                        required=True)

    # --travel-mode parameter
    help_string = ("The name of the travel mode from the network data source that should be used for the analysis. "
                   'If the travel mode name has spaces, enclose the name in double quotes such as "Driving Time"')
    parser.add_argument("-t", "--travel-mode", action="store", dest="travel_mode", help=help_string, required=True)

    # --output-folder parameter
    help_string = "The full catalog path to an existing folder where the tool will create the outputs."
    parser.add_argument("-f", "--folder", action="store", dest="output_folder", help=help_string, required=True)

    # --cutoff parameter
    help_string = ("The impedance value at which to stop searching for destinations for a given origin. This value "
                   "should be specified in Minutes if your chosen travel mode is time based or in Miles if your chosen "
                   "travel mode is distance based. By default no cutoff is used.")
    parser.add_argument("-c", "--cutoff", action="store", type=float, dest="cutoff", help=help_string, default=0)

    # --target-count parameter
    help_string = ("The number of destinations to find per origin. By default, no limit is used, and all destinations"
                   " are found.")
    parser.add_argument("-T", "--target-count", action="store", type=int, dest="target_count", help=help_string,
                        default=0)

    # --workers parameter
    help_string = ("The number of parallel process to use for the analysis. The default is 2. When solving against a "
                   "local network dataset, this number should be set equal to the number of physical CPU cores on your "
                   "machine to achieve maximum performance. When solving against a service, this number should be set "
                   "equal to the number of instances available for the service.")
    parser.add_argument("-w", "--workers", action="store", type=int, dest="workers", help=help_string, default=2)

    # --max-od-size parameter
    max_od_size = 5000000
    help_string = (f"The maximum number of rows that should be present in a single od cost matrix feature class. "
                   "This number governs how many origins and destinations are processed in one iteration. "
                   "For example, if you have 500 origins and 100 destinations and maximum od size is 10,000, "
                   "each iteration will process 100 origins and 100 destinations. The default is {max_od_size}")
    parser.add_argument("-m", "--max-od-size", action="store", type=int, dest="max_od_size", help=help_string,
                        default=max_od_size)

    # Get arguments as dictionary.
    args = vars(parser.parse_args())
    # Convert cutoff and target_count to None if they are not specified.
    cutoff = args["cutoff"]
    target_count = args["target_count"]
    args["cutoff"] = cutoff if cutoff else None
    args["target_count"] = target_count if target_count else None

    # setup the module logger
    ODCostMatrix.setup_logger(logger)

    # call the main execution
    start_time = time.time()
    main(**args)
    logger.info("Completed in %.2f minutes", (time.time() - start_time) / 60)


if __name__ == "__main__":
    _cli()
