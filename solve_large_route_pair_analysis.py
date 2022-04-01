"""Compute a large analysis with origins preassigned to specific destinations
by chunking the inputs and solving in parallel. Write outputs into a single
combined feature class.  TODO

This is a sample script users can modify to fit their specific needs.

This script can be called from the script tool definition or from the command line.

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
import os
import sys
import time
import datetime
import uuid
import traceback
import argparse
import subprocess
from math import floor
from distutils.util import strtobool

import arcpy

import helpers
from rt_config import RT_PROPS  # Import Route settings from config file

arcpy.env.overwriteOutput = True


class RoutePairSolver:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Compute routes between pre-assigned origins and destinations pairs in parallel and combine results.

    This class preprocesses and validates inputs and then spins up a subprocess to do the actual Route
    calculations. This is necessary because the a script tool running in the ArcGIS Pro UI cannot directly call
    multiprocessing using concurrent.futures. We must spin up a subprocess, and the subprocess must spawn parallel
    processes for the calculations. Thus, this class does all the pre-processing, passes inputs to the subprocess, and
    handles messages returned by the subprocess. The subprocess, parallel_route_pairs.py, actually does the
    calculations.
    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, origins, origin_id_field, assigned_dest_field, destinations, dest_id_field,
        network_data_source, travel_mode, time_units, distance_units,
        chunk_size, max_processes, output_routes,
        time_of_day=None, barriers=None, precalculate_network_locations=True, sort_origins=True
    ):
        """Initialize the RoutePairSolver class.

        Args:
        TODO
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
            cutoff (float, optional): Impedance cutoff to limit the Route solve. Interpreted in the time_units
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
        self.origin_id_field = origin_id_field
        self.assigned_dest_field = assigned_dest_field
        self.destinations = destinations
        self.dest_id_field = dest_id_field
        self.network_data_source = network_data_source
        self.travel_mode = travel_mode
        self.time_units = time_units
        self.distance_units = distance_units
        self.chunk_size = chunk_size
        self.max_processes = max_processes
        self.time_of_day = time_of_day
        self.time_of_day_dt = None  # Set during validation
        self.barriers = barriers if barriers else []
        self.should_precalc_network_locations = precalculate_network_locations
        self.should_sort_origins = sort_origins
        self.output_routes = output_routes

        # Scratch folder to store intermediate outputs from the Route processes
        self.unique_id = uuid.uuid4().hex
        self.scratch_folder = os.path.join(arcpy.env.scratchFolder, "rt_" + self.unique_id)  # pylint: disable=no-member
        arcpy.AddMessage(f"Intermediate outputs will be written to {self.scratch_folder}.")
        self.scratch_gdb = os.path.join(self.scratch_folder, "Inputs.gdb")
        self.output_origins = os.path.join(self.scratch_gdb, "Origins")  # pylint: disable=no-member
        self.output_destinations = os.path.join(self.scratch_gdb, "Destinations")  # pylint: disable=no-member

        self.is_service = helpers.is_nds_service(self.network_data_source)
        self.service_limits = None  # Set during validation
        self.is_agol = False  # Set during validation

        self.destination_ids = []  # Populated during validation

    def _validate_inputs(self):
        """Validate the Route inputs."""
        # Validate input numerical values
        if self.chunk_size < 1:
            err = "Chunk size must be greater than 0."
            arcpy.AddError(err)
            raise ValueError(err)
        if self.max_processes < 1:
            err = "Maximum allowed parallel processes must be greater than 0."
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
        self._validate_unique_id_field(self.origins, self.origin_id_field)
        self.destination_ids = self._validate_unique_id_field(self.destinations, self.dest_id_field)
        self._validate_assigned_dest_field()

        # Validate network
        self.network_data_source = helpers.validate_network_data_source(self.network_data_source)

        # Validate Route settings and convert travel mode to a JSON string
        self.travel_mode = self._validate_route_settings()

        # For a services solve, get tool limits and validate max processes and chunk size
        if self.is_service:
            self.service_limits, self.is_agol = helpers.get_tool_limits_and_is_agol(
                self.network_data_source, "asyncRoute", "FindRoutes")
            if self.is_agol:
                self.max_processes = helpers.update_agol_max_processes(self.max_processes)
            self._update_max_inputs_for_service()
            if self.should_precalc_network_locations:
                arcpy.AddWarning(
                    "Cannot precalculate network location fields when the network data source is a service.")
                self.should_precalc_network_locations = False

    @staticmethod
    def _validate_unique_id_field(input_features, id_field):
        """Validate the unique ID field in the input features.

        Args:
            input_features: Input feature class to check.
            id_field: Field in the input features to check.

        Raises:
            ValueError: If the field does not exist in the destinations dataset
            ValueError: If the field values are not unique
        """
        # Check if the field exists
        field_names = [f.name for f in arcpy.ListFields(input_features, wild_card=id_field)]
        if id_field not in field_names:
            err = f"Unique ID field {id_field} does not exist in dataset {input_features}."
            arcpy.AddError(err)
            raise ValueError(err)
        # Populate a list of destination IDs and verify that they are unique
        ids = []
        for row in arcpy.da.SearchCursor(input_features, [id_field]):  # pylint:disable = no-member
            ids.append(row[0])
        num_rows = len(ids)
        ids = list(set(ids))
        if len(ids) != num_rows:
            err = f"Non-unique values were found in the unique ID field {id_field} in {input_features}."
            arcpy.AddError(err)
            raise ValueError(err)
        # Return the list of unique IDs
        return ids

    def _validate_assigned_dest_field(self):
        """Validate the assigned destination field in the origins.

        Raises:
            ValueError: If the field does not exist in the origins dataset
            ValueError: If all origins are assigned to invalid destinations
        """
        # Check if the field exists
        field_names = [f.name for f in arcpy.ListFields(self.origins, wild_card=self.assigned_dest_field)]
        if self.assigned_dest_field not in field_names:
            err = (f"Assigned destination field {self.assigned_dest_field} does not exist in Origins dataset "
                   f"{self.origins}.")
            arcpy.AddError(err)
            raise ValueError(err)
        # Check if there are any origins whose destination IDs don't match up with a known destination
        num_invalid = 0
        num_total = 0
        for row in arcpy.da.SearchCursor(self.origins, [self.assigned_dest_field]):  # pylint: disable=no-member
            num_total += 1
            if row[0] not in self.destination_ids:
                num_invalid += 1
        if num_invalid > 0:
            if num_invalid == num_total:
                err = (f"All origins in the Origins dataset {self.origins} have invalid values in the assigned "
                       f"destination field {self.assigned_dest_field} that do not correspond to values in the "
                       f"destinations unique ID field {self.dest_id_field} in {self.destinations}. Ensure that you "
                       "have chosen the correct datasets and fields and that the field types match.")
                arcpy.AddError(err)
                raise ValueError(err)
            else:
                arcpy.AddWarning((
                    f"{num_invalid} of {num_total} origins have invalid values in the assigned destination field "
                    f"{self.assigned_dest_field} that do not correspond to values in the destinations unique ID field "
                    f"{self.dest_id_field} in {self.destinations}. The origins will be ignored in the analysis."))

    def _validate_route_settings(self):
        """Validate Route settings by spinning up a dummy Route object.

        Raises:
            ValueError: If the travel mode doesn't have a name

        Returns:
            str: JSON string representation of the travel mode
        """
        arcpy.AddMessage("Validating Route settings...")
        # Validate time and distance units
        time_units = helpers.convert_time_units_str_to_enum(self.time_units)
        distance_units = helpers.convert_distance_units_str_to_enum(self.distance_units)
        # Create a dummy Route object and set properties
        try:
            rt = arcpy.nax.Route(self.network_data_source)
            rt.travelMode = self.travel_mode
            rt.timeUnits = time_units
            rt.distanceUnits = distance_units
            rt.timeOfDay = self.time_of_day_dt
        except Exception:
            arcpy.AddError("Invalid Route settings.")
            errs = traceback.format_exc().splitlines()
            for err in errs:
                arcpy.AddError(err)
            raise

        # Return a JSON string representation of the travel mode to pass to the subprocess
        return rt.travelMode._JSON  # pylint: disable=protected-access

    def _update_max_inputs_for_service(self):
        """Check the user's specified max origins and destinations and reduce max to portal limits if required."""
        lim_max_stops = self.service_limits["maximumStops"]
        if lim_max_stops:
            lim_max_stops = int(lim_max_stops)
            # There will be two stops in each route. Set our chunk size to the max stop limit / 2.
            if lim_max_stops < self.chunk_size * 2:
                self.chunk_size = floor(lim_max_stops / 2)
                arcpy.AddMessage(
                    f"Max OD pairs per chunk has been updated to {self.chunk_size} to accommodate service limits.")

    def _sort_origins_by_assigned_destination(self):
        """Sort the origins by the assigned destination field.

        Also adds a field called "OriginOID" to the input feature class to preserve the original OID values.
        """
        arcpy.AddMessage("Sorting origins by assigned destination...")

        # Sort input features
        sorted_origins = self.output_origins + "_Sorted"
        arcpy.management.Sort(self.output_origins, sorted_origins, [[self.assigned_dest_field, "ASCENDING"]])
        self.output_origins = sorted_origins

    def _make_field_mappings(self, input_fc, oid_field_name):
        """Make field mappings for use in FeatureClassToFeatureClass to transfer original ObjectID.

        Args:
            input_fc (str, layer): Input feature class or layer
            oid_field_name (str): ObjectID field name of the input_fc

        Returns:
            (arcpy.FieldMappings, str): Field mappings for use in FeatureClassToFeatureClass that maps the ObjectID
                field to a unique new field name so its values will be preserved after copying the feature class. The
                new unique field name.
        """
        field_mappings = arcpy.FieldMappings()
        field_mappings.addTable(input_fc)
        # Create a new output field with a unique name to store the original OID
        new_field = arcpy.Field()
        new_field_name = f"OID_{self.unique_id}"
        new_field.name = new_field_name
        new_field.aliasName = "Original Unique ID"
        new_field.type = "Integer"
        # Create a new field map object and map the ObjectID to the new output field
        new_fm = arcpy.FieldMap()
        new_fm.addInputField(input_fc, oid_field_name)
        new_fm.outputField = new_field
        # Add the new field map
        field_mappings.addFieldMap(new_fm)
        return field_mappings, new_field_name

    def _preprocess_inputs(self):
        """Preprocess the input feature classes to prepare them for use in the Route."""
        # Make scratch folder and geodatabase
        os.mkdir(self.scratch_folder)
        arcpy.management.CreateFileGDB(os.path.dirname(self.scratch_gdb), os.path.basename(self.scratch_gdb))

        arcpy.AddMessage("Copying input origins and destinations to outputs...")

        # Copy Origins to output
        origins_oid_field = arcpy.Describe(self.origins).oidFieldName
        field_mappings = None
        if self.origin_id_field == origins_oid_field:
            field_mappings, self.origin_id_field = self._make_field_mappings(self.origins, origins_oid_field)
        arcpy.conversion.FeatureClassToFeatureClass(
            self.origins,
            os.path.dirname(self.output_origins),
            os.path.basename(self.output_origins),
            field_mapping=field_mappings
        )

        # Copy Destinations to output
        dest_oid_field = arcpy.Describe(self.destinations).oidFieldName
        field_mappings = None
        if self.dest_id_field == dest_oid_field:
            field_mappings, self.dest_id_field = self._make_field_mappings(self.destinations, dest_oid_field)
        arcpy.conversion.FeatureClassToFeatureClass(
            self.destinations,
            os.path.dirname(self.output_destinations),
            os.path.basename(self.output_destinations),
            field_mapping=field_mappings
        )

        # Sort origins by assigned destination
        if self.should_sort_origins:
            self._sort_origins_by_assigned_destination()

        # Precalculate network location fields for inputs
        if not self.is_service and self.should_precalc_network_locations:
            helpers.precalculate_network_locations(
                self.output_destinations, self.network_data_source, self.travel_mode, RT_PROPS)
            for barrier_fc in self.barriers:
                helpers.precalculate_network_locations(
                    barrier_fc, self.network_data_source, self.travel_mode, RT_PROPS)

    def _execute_solve(self):
        """Solve the multi-route analysis."""
        # Launch the parallel_route_pairs script as a subprocess so it can spawn parallel processes. We have to do this
        # because a tool running in the Pro UI cannot call concurrent.futures without opening multiple instances of Pro.
        cwd = os.path.dirname(os.path.abspath(__file__))
        rt_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(cwd, "parallel_route_pairs.py"),
            "--origins", self.output_origins,
            "--origins-id-field", self.origin_id_field,
            "--assigned-dest-field", self.assigned_dest_field,
            "--destinations", self.output_destinations,
            "--destinations-id-field", self.dest_id_field,  ## TODO: If ID field is ObjectID, transfer it
            "--network-data-source", self.network_data_source,
            "--travel-mode", self.travel_mode,
            "--time-units", self.time_units,
            "--distance-units", self.distance_units,
            "--max-routes", str(self.chunk_size),
            "--max-processes", str(self.max_processes),
            "--out-routes", str(self.output_routes),
            "--scratch-folder", self.scratch_folder
        ]
        # Include other optional parameters if relevant
        if self.barriers:
            rt_inputs += ["--barriers"]
            rt_inputs += self.barriers
        if self.time_of_day:
            rt_inputs += ["--time-of-day", self.time_of_day]

        # We do not want to show the console window when calling the command line tool from within our GP tool.
        # This can be done by setting this hex code.
        create_no_window = 0x08000000
        with subprocess.Popen(
            rt_inputs,
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
                time.sleep(.1)

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
                arcpy.AddError("Route script failed.")

    def solve_large_route_pair_analysis(self):
        """Solve the large multi-route with known OD pairs in parallel."""
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

    # --origins-id-field parameter
    help_string = "The name of the unique ID field in origins."
    parser.add_argument(
        "-oif", "--origins-id-field", action="store", dest="origin_id_field", help=help_string, required=True)

    # --assigned-dest-field parameter
    help_string = "The name of the field in origins indicating the assigned destination."
    parser.add_argument(
        "-adf", "--assigned-dest-field", action="store", dest="assigned_dest_field", help=help_string, required=True)

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
        "-mr", "--max-routes", action="store", dest="chunk_size", type=int, help=help_string, required=True)

    # --max-processes parameter
    help_string = "Maximum number parallel processes to use for the Route solves."
    parser.add_argument(
        "-mp", "--max-processes", action="store", dest="max_processes", type=int, help=help_string, required=True)

    # --out-routes parameter
    help_string = "The full catalog path to the output routes feature class."
    parser.add_argument("-r", "--out-routes", action="store", dest="output_routes", help=help_string, required=True)

    # --time-of-day parameter
    help_string = (f"The time of day for the analysis. Must be in {helpers.DATETIME_FORMAT} format. Set to None for "
                   "time neutral.")
    parser.add_argument("-tod", "--time-of-day", action="store", dest="time_of_day", help=help_string, required=False)

    # --barriers parameter
    help_string = "A list of catalog paths to the feature classes containing barriers to use in the Route."
    parser.add_argument(
        "-b", "--barriers", action="store", dest="barriers", help=help_string, nargs='*', required=False)

    # --precalculate-network-locations parameter
    help_string = "Whether or not to precalculate network location fields before solving the analysis."
    parser.add_argument(
        "-pnl", "--precalculate-network-locations", action="store", type=lambda x: bool(strtobool(x)),
        dest="precalculate_network_locations", help=help_string, required=True)

    # --sort-origins parameter
    help_string = "Whether or not to sort the origins by assigned destination before solving the analysis."
    parser.add_argument(
        "-so", "--sort-origins", action="store", type=lambda x: bool(strtobool(x)),
        dest="sort_origins", help=help_string, required=True)

    # Get arguments as dictionary.
    args = vars(parser.parse_args())

    # Solve the Route
    rt_solver = RoutePairSolver(**args)
    rt_solver.solve_large_route_pair_analysis()


if __name__ == "__main__":
    # Run script from the command line
    _run_from_command_line()
