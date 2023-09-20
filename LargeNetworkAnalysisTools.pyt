"""Python toolbox that defines tools for solving large network analysis problems.

This is a sample script users can modify to fit their specific needs.

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
# Disable a bunch of linter errors caused by the standard python toolbox class definitions that we cannot change
# pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods, too-many-locals
# pylint: disable=useless-return, unused-argument
import os
from os import cpu_count
import arcpy

import helpers
from solve_large_odcm import ODCostMatrixSolver
from solve_large_route_pair_analysis import RoutePairSolver


class Toolbox(object):
    """Tools for solving large network analysis problems."""

    def __init__(self):
        """Define the toolbox."""
        self.label = "Large Network Analysis Tools"
        self.alias = "LargeNetworkAnalysisTools"

        # List of tool classes associated with this toolbox
        self.tools = [SolveLargeODCostMatrix, SolveLargeAnalysisWithKnownPairs, ParallelCalculateLocations]


class SolveLargeODCostMatrix(object):
    """Sample script tool to solve a large OD Cost Matrix by chunking the input data and solving in parallel."""

    def __init__(self):
        """Define the tool."""
        self.label = "Solve Large OD Cost Matrix"
        self.description = (
            "Solve a large OD Cost Matrix problem by chunking the input data and solving in parallel."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_origins = arcpy.Parameter(
            displayName="Origins",
            name="Origins",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_destinations = arcpy.Parameter(
            displayName="Destinations",
            name="Destinations",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_network = arcpy.Parameter(
            displayName="Network Data Source",
            name="Network_Data_Source",
            datatype="GPNetworkDataSource",
            parameterType="Required",
            direction="Input"
        )

        param_travel_mode = arcpy.Parameter(
            displayName="Travel Mode",
            name="Travel_Mode",
            datatype="NetworkTravelMode",
            parameterType="Required",
            direction="Input"
        )
        param_travel_mode.parameterDependencies = [param_network.name]

        param_time_units = arcpy.Parameter(
            displayName="Time Units",
            name="Time_Units",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_time_units.filter.list = helpers.TIME_UNITS
        param_time_units.value = "Minutes"

        param_distance_units = arcpy.Parameter(
            displayName="Distance Units",
            name="Distance_Units",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_distance_units.filter.list = helpers.DISTANCE_UNITS
        param_distance_units.value = "Miles"

        param_chunk_size = arcpy.Parameter(
            displayName="Maximum Origins and Destinations per Chunk",
            name="Max_Inputs_Per_Chunk",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_chunk_size.value = 1000

        param_max_processes = arcpy.Parameter(
            displayName="Maximum Number of Parallel Processes",
            name="Max_Processes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_max_processes.value = 4

        param_out_origins = arcpy.Parameter(
            displayName="Output Updated Origins",
            name="Output_Updated_Origins",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

        param_out_destinations = arcpy.Parameter(
            displayName="Output Updated Destinations",
            name="Output_Updated_Destinations",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

        param_output_format = arcpy.Parameter(
            displayName="Output OD Cost Matrix Format",
            name="Output_Format",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_output_format.filter.list = helpers.OUTPUT_FORMATS
        param_output_format.value = helpers.OUTPUT_FORMATS[0]

        param_out_od_lines = arcpy.Parameter(
            displayName="Output OD Lines Feature Class",
            name="Output_OD_Lines_Feature_Class",
            datatype="DEFeatureClass",
            parameterType="Optional",
            direction="Output"
        )

        param_out_folder = arcpy.Parameter(
            displayName="Output Folder",
            name="Output_Folder",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Output"
        )

        param_cutoff = arcpy.Parameter(
            displayName="Cutoff",
            name="Cutoff",
            datatype="GPDouble",
            parameterType="Optional",
            direction="Input"
        )

        param_num_dests = arcpy.Parameter(
            displayName="Number of Destinations to Find for Each Origin",
            name="Num_Destinations",
            datatype="GPLong",
            parameterType="Optional",
            direction="Input"
        )

        param_time_of_day = arcpy.Parameter(
            displayName="Time of Day",
            name="Time_Of_Day",
            datatype="GPDate",
            parameterType="Optional",
            direction="Input"
        )

        param_barriers = arcpy.Parameter(
            displayName="Barriers",
            name="Barriers",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            category="Advanced"
        )

        param_precalculate_network_locations = arcpy.Parameter(
            displayName="Precalculate Network Locations",
            name="Precalculate_Network_Locations",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced"
        )
        param_precalculate_network_locations.value = True

        param_sort_inputs = arcpy.Parameter(
            displayName="Spatially Sort Inputs",
            name="Sort_Inputs",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced"
        )
        param_sort_inputs.value = True

        params = [
            param_origins,  # 0
            param_destinations,  # 1
            param_network,  # 2
            param_travel_mode,  # 3
            param_time_units,  # 4
            param_distance_units,  # 5
            param_chunk_size,  # 6
            param_max_processes,  # 7
            param_out_origins,  # 8
            param_out_destinations,  # 9
            param_output_format,  # 10
            param_out_od_lines,  # 11
            param_out_folder,  # 12
            param_cutoff,  # 13
            param_num_dests,  # 14
            param_time_of_day,  # 15
            param_barriers,  # 16
            param_precalculate_network_locations,  # 17
            param_sort_inputs  # 18
        ]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        param_network = parameters[2]
        param_output_format = parameters[10]
        param_out_od_lines = parameters[11]
        param_out_folder = parameters[12]
        param_precalculate = parameters[17]
        param_sort = parameters[18]

        # Make appropriate OD Cost Matrix output parameter visible based on chosen output format
        if not param_output_format.hasBeenValidated and param_output_format.altered and param_output_format.valueAsText:
            try:
                output_format = helpers.convert_output_format_str_to_enum(param_output_format.valueAsText)
                if output_format is helpers.OutputFormat.featureclass:
                    param_out_od_lines.enabled = True
                    param_out_folder.enabled = False
                else:  # For CSV and Arrow
                    param_out_od_lines.enabled = False
                    param_out_folder.enabled = True
            except ValueError:
                # The output format is invalid. Just do nothing.
                pass

        # Turn off and hide Precalculate Network Locations parameter if the network data source is a service
        update_precalculate_parameter(param_network, param_precalculate)

        # Turn off and hide the Sort Inputs parameter if the Advanced license is not available.
        if helpers.arc_license != "ArcInfo":
            param_sort.value = False
            param_sort.enabled = False

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        param_network = parameters[2]
        param_chunk_size = parameters[6]
        param_max_processes = parameters[7]
        param_output_format = parameters[10]
        param_out_od_lines = parameters[11]
        param_out_folder = parameters[12]

        # Give a warning for very large chunk sizes
        if param_chunk_size.altered and param_chunk_size.valueAsText:
            if int(param_chunk_size.valueAsText) > 15000:
                param_chunk_size.setWarningMessage((
                    "The tool will likely run faster with smaller chunk sizes. Consult the user's guide for "
                    "recommendations."
                ))

        # Validate max processes and cap it when necessary
        cap_max_processes(param_network, param_max_processes)
        # If the network data source is a service, show an error if attempting to use Arrow output, which is not
        # supported at this time
        if param_network.altered and param_network.valueAsText and helpers.is_nds_service(param_network.valueAsText):
            if param_output_format.altered and param_output_format.valueAsText:
                output_format = helpers.convert_output_format_str_to_enum(param_output_format.valueAsText)
                if output_format is helpers.OutputFormat.arrow:
                    param_output_format.setErrorMessage((
                        f"{param_output_format.valueAsText} output format is not available when a service is used as "
                        "the network data source."
                    ))

        # Make the appropriate output parameter required based on the user's choice of output format. Just require
        # whichever output parameter is enabled. Enablement is controlled in updateParameters() based on the user's
        # choice in the output format parameter.
        # The 735 error code doesn't display an actual error but displays the little red star to indicate that the
        # parameter is required.
        for param in [param_out_od_lines, param_out_folder]:
            if param.enabled:
                if not param.valueAsText:
                    param.setIDMessage("Error", 735, param.displayName)
            else:
                param.clearMessage()

        # Do not allow the user to overwrite an existing output folder. This prevents them from destroying their project
        # folder or C: drive, etc.
        if param_out_folder.enabled:
            out_folder = param_out_folder.valueAsText
            if out_folder and os.path.exists(out_folder):
                # <out_folder> already exists
                param_out_folder.setIDMessage("Error", 12, out_folder)

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Initialize the solver class
        time_of_day = parameters[15].value
        if time_of_day:
            time_of_day = time_of_day.strftime(helpers.DATETIME_FORMAT)
        od_solver = ODCostMatrixSolver(
            parameters[0].value,  # origins
            parameters[1].value,  # destinations
            get_catalog_path(parameters[2]),  # network
            parameters[3].value,  # travel mode
            parameters[8].valueAsText,  # output origins
            parameters[9].valueAsText,  # output destinations
            parameters[6].value,  # chunk size
            parameters[7].value,  # max processes
            parameters[4].valueAsText,  # time units
            parameters[5].valueAsText,  # distance units
            parameters[10].valueAsText,  # output format
            parameters[11].valueAsText,  # output OD lines
            parameters[12].valueAsText,  # output data folder
            parameters[13].value,  # cutoff
            parameters[14].value,  # number of destinations to find
            time_of_day,  # time of day
            parameters[17].value,  # Should precalculate network locations
            parameters[18].value,  # Should spatially sort inputs
            get_catalog_path_multivalue(parameters[16])  # barriers
        )

        # Solve the OD Cost Matrix analysis
        od_solver.solve_large_od_cost_matrix()

        return


class SolveLargeAnalysisWithKnownPairs(object):
    """Sample script tool to solve a large analysis with known origin-destination pairs."""

    def __init__(self):
        """Define the tool."""
        self.label = "Solve Large Analysis With Known OD Pairs"
        self.description = ((
            "Solve a large analysis with known origin-destination pairs by chunking the input data and solving in "
            "parallel."
        ))
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_origins = arcpy.Parameter(
            displayName="Origins",
            name="Origins",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_origin_id_field = arcpy.Parameter(
            displayName="Origin Unique ID Field",
            name="Origin_Unique_ID_Field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param_origin_id_field.parameterDependencies = [param_origins.name]
        param_origin_id_field.filter.list = helpers.ID_FIELD_TYPES

        param_destinations = arcpy.Parameter(
            displayName="Destinations",
            name="Destinations",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )

        param_dest_id_field = arcpy.Parameter(
            displayName="Destination Unique ID Field",
            name="Destination_Unique_ID_Field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param_dest_id_field.parameterDependencies = [param_destinations.name]
        param_dest_id_field.filter.list = helpers.ID_FIELD_TYPES

        param_pair_type = arcpy.Parameter(
            displayName="Origin-Destination Assignment Type",
            name="OD_Pair_Type",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_pair_type.filter.list = helpers.PAIR_TYPES
        param_pair_type.value = helpers.PAIR_TYPES[0]

        param_assigned_dest_field = arcpy.Parameter(
            displayName="Assigned Destination Field",
            name="Assigned_Destination_Field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param_assigned_dest_field.parameterDependencies = [param_origins.name]
        param_assigned_dest_field.filter.list = helpers.ID_FIELD_TYPES

        param_pair_table = arcpy.Parameter(
            displayName="Origin-Destination Pair Table",
            name="OD_Pair_Table",
            datatype="GPTableView",
            parameterType="Required",
            direction="Input"
        )

        param_pair_table_origin_id_field = arcpy.Parameter(
            displayName="Origin ID Field in Origin-Destination Pair Table",
            name="Pair_Table_Origin_Unique_ID_Field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param_pair_table_origin_id_field.parameterDependencies = [param_pair_table.name]
        param_pair_table_origin_id_field.filter.list = helpers.ID_FIELD_TYPES

        param_pair_table_dest_id_field = arcpy.Parameter(
            displayName="Destination ID Field in Origin-Destination Pair Table",
            name="Pair_Table_Destination_Unique_ID_Field",
            datatype="Field",
            parameterType="Required",
            direction="Input"
        )
        param_pair_table_dest_id_field.parameterDependencies = [param_pair_table.name]
        param_pair_table_dest_id_field.filter.list = helpers.ID_FIELD_TYPES

        param_network = arcpy.Parameter(
            displayName="Network Data Source",
            name="Network_Data_Source",
            datatype="GPNetworkDataSource",
            parameterType="Required",
            direction="Input"
        )

        param_travel_mode = arcpy.Parameter(
            displayName="Travel Mode",
            name="Travel_Mode",
            datatype="NetworkTravelMode",
            parameterType="Required",
            direction="Input"
        )
        param_travel_mode.parameterDependencies = [param_network.name]

        param_time_units = arcpy.Parameter(
            displayName="Time Units",
            name="Time_Units",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_time_units.filter.list = helpers.TIME_UNITS
        param_time_units.value = "Minutes"

        param_distance_units = arcpy.Parameter(
            displayName="Distance Units",
            name="Distance_Units",
            datatype="GPString",
            parameterType="Required",
            direction="Input"
        )
        param_distance_units.filter.list = helpers.DISTANCE_UNITS
        param_distance_units.value = "Miles"

        param_chunk_size = arcpy.Parameter(
            displayName="Maximum OD Pairs per Chunk",
            name="Max_Pairs_Per_Chunk",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_chunk_size.value = 1000

        param_max_processes = arcpy.Parameter(
            displayName="Maximum Number of Parallel Processes",
            name="Max_Processes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_max_processes.value = 4

        param_out_routes = arcpy.Parameter(
            displayName="Output Routes",
            name="Output_Routes",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

        param_time_of_day = arcpy.Parameter(
            displayName="Time of Day",
            name="Time_Of_Day",
            datatype="GPDate",
            parameterType="Optional",
            direction="Input"
        )

        param_barriers = arcpy.Parameter(
            displayName="Barriers",
            name="Barriers",
            datatype="GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
            multiValue=True,
            category="Advanced"
        )

        param_precalculate_network_locations = arcpy.Parameter(
            displayName="Precalculate Network Locations",
            name="Precalculate_Network_Locations",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced"
        )
        param_precalculate_network_locations.value = True

        param_sort_origins = arcpy.Parameter(
            displayName="Sort Origins by Assigned Destination",
            name="Sort_Origins",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced"
        )
        param_sort_origins.value = True

        param_reverse_direction = arcpy.Parameter(
            displayName="Reverse Direction of Travel",
            name="Reverse_Direction",
            datatype="GPBoolean",
            parameterType="Optional",
            direction="Input",
            category="Advanced"
        )
        param_reverse_direction.value = False

        params = [
            param_origins,  # 0
            param_origin_id_field,  # 1
            param_destinations,  # 2
            param_dest_id_field,  # 3
            param_pair_type,  # 4
            param_assigned_dest_field,  # 5
            param_pair_table,  # 6
            param_pair_table_origin_id_field,  # 7
            param_pair_table_dest_id_field,  # 8
            param_network,  # 9
            param_travel_mode,  # 10
            param_time_units,  # 11
            param_distance_units,  # 12
            param_chunk_size,  # 13
            param_max_processes,  # 14
            param_out_routes,  # 15
            param_time_of_day,  # 16
            param_barriers,  # 17
            param_precalculate_network_locations,  # 18
            param_sort_origins,  # 19
            param_reverse_direction  # 20
        ]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        param_network = parameters[9]
        param_precalculate = parameters[18]
        param_pair_type = parameters[4]
        param_assigned_dest_field = parameters[5]
        param_pair_table = parameters[6]
        param_pair_table_origin_id_field = parameters[7]
        param_pair_table_dest_id_field = parameters[8]
        param_sort_origins = parameters[19]
        param_reverse_direction = parameters[20]

        # Turn off and hide Precalculate Network Locations parameter if the network data source is a service
        update_precalculate_parameter(param_network, param_precalculate)

        # Toggle parameter visibility based on Origin-Destination Assignment Type
        if not param_pair_type.hasBeenValidated and param_pair_type.altered and param_pair_type.valueAsText:
            try:
                pair_type = helpers.convert_pair_type_str_to_enum(param_pair_type.valueAsText)
                if pair_type is helpers.PreassignedODPairType.one_to_one:
                    # Enable parameters associated with one-to-one
                    param_assigned_dest_field.enabled = True
                    param_sort_origins.enabled = True
                    param_reverse_direction.enabled = True
                    # Disable parameter associated with other pair types and reset their values
                    param_pair_table.value = None
                    param_pair_table.enabled = False
                    param_pair_table_origin_id_field.value = None
                    param_pair_table_origin_id_field.enabled = False
                    param_pair_table_dest_id_field.value = None
                    param_pair_table_dest_id_field.enabled = False
                elif pair_type is helpers.PreassignedODPairType.many_to_many:
                    # Enable parameter associated with many-to-many
                    param_pair_table.enabled = True
                    param_pair_table_origin_id_field.enabled = True
                    param_pair_table_dest_id_field.enabled = True
                    # Disable parameter associated with other pair types and reset their values
                    param_assigned_dest_field.enabled = False
                    param_assigned_dest_field.value = None
                    param_sort_origins.enabled = False
                    param_sort_origins.value = False
                    param_reverse_direction.enabled = False
                    param_reverse_direction.value = False
            except Exception:  # pylint: disable=broad-except
                # Invalid pair type.  Don't modify any parameters.
                pass

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        param_network = parameters[9]
        param_max_processes = parameters[14]
        param_assigned_dest_field = parameters[5]
        param_pair_table = parameters[6]
        param_pair_table_origin_id_field = parameters[7]
        param_pair_table_dest_id_field = parameters[8]

        # Validate max processes and cap it when necessary
        cap_max_processes(param_network, param_max_processes)

        # Make the appropriate OD pair parameters required based on the user's choice of Origin-Destination Assignment
        # Type. Just require whichever parameters is enabled. Enablement is controlled in updateParameters() based on
        # the user's choice in the Origin-Destination Assignment Type parameter.
        # The 735 error code doesn't display an actual error but displays the little red star to indicate that the
        # parameter is required.
        for param in [
            param_assigned_dest_field,
            param_pair_table,
            param_pair_table_origin_id_field,
            param_pair_table_dest_id_field
        ]:
            if param.enabled:
                if not param.valueAsText:
                    param.setIDMessage("Error", 735, param.displayName)
            else:
                param.clearMessage()

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Initialize the solver class
        time_of_day = parameters[16].value
        if time_of_day:
            time_of_day = time_of_day.strftime(helpers.DATETIME_FORMAT)
        rt_solver = RoutePairSolver(
            parameters[0].value,  # origins
            parameters[1].valueAsText,  # unique origin ID field
            parameters[2].value,  # destinations
            parameters[3].valueAsText,  # unique destination ID field
            helpers.convert_pair_type_str_to_enum(parameters[4].valueAsText),  # pair type
            get_catalog_path(parameters[9]),  # network
            parameters[10].value,  # travel mode
            parameters[11].valueAsText,  # time units
            parameters[12].valueAsText,  # distance units
            parameters[13].value,  # chunk size
            parameters[14].value,  # max processes
            parameters[15].valueAsText,  # output routes
            parameters[5].valueAsText,  # assigned destination field
            parameters[6].value,  # pair table
            parameters[7].valueAsText,  # pair table origin ID field
            parameters[8].valueAsText,  # pair table destination ID field
            time_of_day,  # time of day
            get_catalog_path_multivalue(parameters[17]),  # barriers
            parameters[18].value,  # Should precalculate network locations
            parameters[19].value,  # Should sort origins
            parameters[20].value  # Reverse direction of travel
        )

        # Solve the OD Cost Matrix analysis
        rt_solver.solve_large_route_pair_analysis()

        return


class ParallelCalculateLocations(object):
    """Sample script tool to calculate locations for a large dataset in parallel."""

    def __init__(self):
        """Define the tool."""
        self.label = "Parallel Calculate Locations"
        self.description = "Calculate network locations for a large dataset by chunking it and solving in parallel."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""

        param_in_features = arcpy.Parameter(
            displayName="Input Features",
            name="Input_Features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        param_in_features.filter.list = ["Point"]

        param_out_features = arcpy.Parameter(
            displayName="Output Features",
            name="Output_Features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Output"
        )

        param_network = arcpy.Parameter(
            displayName="Network Dataset",
            name="Network_Dataset",
            datatype="GPNetworkDatasetLayer",
            parameterType="Required",
            direction="Input"
        )

        param_chunk_size = arcpy.Parameter(
            displayName="Maximum Features per Chunk",
            name="Max_Features_Per_Chunk",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_chunk_size.value = 1000

        param_max_processes = arcpy.Parameter(
            displayName="Maximum Number of Parallel Processes",
            name="Max_Processes",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param_max_processes.value = 4

        param_travel_mode = arcpy.Parameter(
            displayName="Travel Mode",
            name="Travel_Mode",
            datatype="NetworkTravelMode",
            parameterType="Optional",
            direction="Input"
        )
        param_travel_mode.parameterDependencies = [param_network.name]

        param_search_tolerance = arcpy.Parameter(
            displayName="Search Tolerance",
            name="Search_Tolerance",
            datatype="GPLinearUnit",
            parameterType="Optional",
            direction="Input"
        )
        param_search_tolerance.value = "5000 Meters"

        param_search_criteria = arcpy.Parameter(
            displayName="Search Criteria",
            name="Search_Criteria",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        # Causes the parameter to be a list of checkboxes instead of a standard multivalue
        param_search_criteria.controlCLSID = "{38C34610-C7F7-11D5-A693-0008C711C8C1}"

        param_search_query = arcpy.Parameter(
            displayName="Search Query",
            name="Search_Query",
            datatype="GPValueTable",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        param_search_query.columns = [
            ['GPString', 'Name'],
            ['GPString', 'Query']
        ]
        param_search_query.filters[0].type = 'ValueList'

        params = [
            param_in_features,  # 0
            param_out_features,  # 1
            param_network,  # 2
            param_chunk_size,  # 3
            param_max_processes,  # 4
            param_travel_mode,  # 5
            param_search_tolerance,  # 6
            param_search_criteria,  # 7
            param_search_query  # 8
        ]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        param_network = parameters[2]
        param_search_criteria = parameters[7]
        param_search_query = parameters[8]

        # Populate available network sources in the search criteria and search query parameters
        if not param_network.hasBeenValidated and param_network.altered and param_network.valueAsText:
            try:
                network = param_network.valueAsText
                source_names = helpers.get_locatable_network_source_names(network)
                default_source_names = helpers.get_default_locatable_network_source_names(network)
                param_search_criteria.filter.list = source_names
                param_search_criteria.value = default_source_names
                param_search_query.filters[0].list = source_names
            except Exception:  # pylint: disable=broad-except
                # Something went wrong in checking the network's sources.  Don't modify any parameters.
                pass

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        param_network = parameters[2]
        param_search_criteria = parameters[7]
        param_search_query = parameters[8]

        # Validate that at least one source is selected for search criteria
        if not param_search_criteria.hasBeenValidated and param_network.valueAsText:
            if not param_search_criteria.valueAsText:
                param_search_criteria.setErrorMessage("At least one network source is required.")

        # Validate no duplicate entries in search query
        if not param_search_query.hasBeenValidated and param_search_query.altered and \
                param_search_query.valueAsText and param_network.valueAsText:
            validate_search_query_param(param_search_query, param_network)

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Construct string-based travel mode
        travel_mode = parameters[5].value
        if travel_mode:
            travel_mode = travel_mode._JSON  # pylint: disable=protected-access
        else:
            travel_mode = ""

        # Construct the search criteria from the selected sources
        source_names = parameters[7].filter.list
        sources_to_locate_on = parameters[7].values
        search_criteria = helpers.construct_search_criteria_string(sources_to_locate_on, source_names)

        # Construct string-based search query
        search_query = parameters[8].valueAsText
        if not search_query:
            search_query = ""

        # Make a temporary copy of the input features to take care of any selection sets and definition queries and to
        # add a field preserving the ObjectID.
        input_features = parameters[0].value
        desc = arcpy.Describe(input_features)
        # Create a unique output field name to preserve the original OID
        in_fields = [f.name for f in desc.fields]
        base_oid_field = "ORIG_OID"
        out_oid_field = base_oid_field
        if out_oid_field in in_fields:
            i = 1
            while out_oid_field in in_fields:
                out_oid_field = base_oid_field + str(i)
                i += 1
        field_mappings = helpers.make_oid_preserving_field_mappings(
            input_features, desc.oidFieldName, out_oid_field)
        temp_inputs = arcpy.CreateUniqueName("TempCLInputs", arcpy.env.scratchGDB)  # pylint:disable = no-member
        arcpy.conversion.FeatureClassToFeatureClass(
            input_features,
            arcpy.env.scratchGDB,  # pylint:disable = no-member
            os.path.basename(temp_inputs),
            field_mapping=field_mappings
        )

        try:
            cl_inputs = [
                "--input-features", temp_inputs,
                "--output-features", parameters[1].valueAsText,
                "--network-data-source", get_catalog_path(parameters[2]),
                "--chunk-size", parameters[3].valueAsText,
                "--max-processes", parameters[4].valueAsText,
                "--travel-mode", travel_mode,
                "--search-tolerance", parameters[6].valueAsText,
                "--search-criteria", search_criteria,
                "--search-query", search_query
            ]
            helpers.execute_subprocess("parallel_calculate_locations.py", cl_inputs)

        finally:
            # Clean up. Delete temporary copy of inputs
            arcpy.management.Delete([temp_inputs])

        return


def get_catalog_path(param):
    """Get the catalog path for a single value parameter if possible.

    Args:
        param (arcpy.Parameter): Parameter from which to retrieve the catalog path.

    Returns:
        str: Catalog path to the data
    """
    param_value = param.value
    if not param_value:
        return ""
    # If the value is a layer object, get its data source (catalog path)
    if hasattr(param_value, "dataSource"):
        catalog_path = param_value.dataSource
        if "DB_CONNECTION_PROPERTIES" in catalog_path:
            # Handle weird SDE or MMPK layers where the .dataSource property returns extra garbage
            catalog_path = arcpy.Describe(param_value).catalogPath
        return catalog_path
    # Otherwise, it's probably already a string catalog path. Just return its text value.
    return param.valueAsText


def get_catalog_path_multivalue(param):
    """Get a list of catalog paths for a multivalue feature layer parameter if possible.

    Args:
        param (arcpy.Parameter): Parameter from which to retrieve the catalog paths.

    Returns:
        list(str): List of catalog paths to the data
    """
    if not param.values:
        return []
    # Get both the values as geoprocessing objects and the string values
    values = param.values
    # Have to strip the quotes that get added if there are spaces in the filepath.
    string_values = [val.strip("'") for val in param.valueAsText.split(";")]
    catalog_paths = []
    for idx, val in enumerate(values):
        # If the value is a layer object, get its data source (catalog path)
        if hasattr(val, "dataSource"):
            catalog_path = val.dataSource
            if "DB_CONNECTION_PROPERTIES" in catalog_path:
                # Handle weird SDE or MMPK layers where the .dataSource property returns extra garbage
                catalog_path = arcpy.Describe(val).catalogPath
            catalog_paths.append(catalog_path)
        # Otherwise, it's probably already a string catalog path. The only way to get it is to retrieve it from the
        # valueAsText string that we split up above.
        else:
            catalog_paths.append(string_values[idx])
    return catalog_paths


def update_precalculate_parameter(param_network, param_precalculate):
    """Turn off and hide Precalculate Network Locations parameter if the network data source is a service.

    Args:
        param_network (arcpy.Parameter): Parameter for the network data source
        param_precalculate (arcpy.Parameter): Parameter for precalculate network locations
    """
    if not param_network.hasBeenValidated and param_network.altered and param_network.valueAsText:
        if helpers.is_nds_service(param_network.valueAsText):
            param_precalculate.value = False
            param_precalculate.enabled = False
        else:
            param_precalculate.enabled = True


def cap_max_processes(param_network, param_max_processes):
    """Validate max processes and cap it when required.

    Args:
        param_network (arcpy.Parameter): Parameter for the network data source
        param_max_processes (arcpy.Parameter): Parameter for the max processes
    """
    if param_max_processes.altered and param_max_processes.valueAsText:
        max_processes = param_max_processes.value
        # Don't allow 0 or negative numbers
        if max_processes <= 0:
            param_max_processes.setErrorMessage("The maximum number of parallel processes must be positive.")
            return
        # Cap max processes to the limit allowed by the concurrent.futures module
        if max_processes > helpers.MAX_ALLOWED_MAX_PROCESSES:
            param_max_processes.setErrorMessage((
                f"The maximum number of parallel processes cannot exceed {helpers.MAX_ALLOWED_MAX_PROCESSES:} due "
                "to limitations imposed by Python's concurrent.futures module."
            ))
            return
        # If the network data source is arcgis.com, cap max processes
        if (
            max_processes > helpers.MAX_AGOL_PROCESSES and
            param_network.altered and
            param_network.valueAsText and
            helpers.is_nds_service(param_network.valueAsText) and
            "arcgis.com" in param_network.valueAsText
        ):
            param_max_processes.setErrorMessage((
                f"The maximum number of parallel processes cannot exceed {helpers.MAX_AGOL_PROCESSES} when the "
                "ArcGIS Online service is used as the network data source."
            ))
            return
        # If the network data source is mgdb and the user has put more processes than the max recommended for
        # the mgdb solvers, set a warning.
        if (
            max_processes > helpers.MAX_RECOMMENDED_MGDB_PROCESSES and
            param_network.altered and
            param_network.valueAsText and
            not helpers.is_nds_service(param_network.valueAsText)
        ):
            try:
                network_obj = param_network.value
                is_mgdb = False
                if hasattr(network_obj, "connectionProperties"):
                    # A network dataset layer in a map typically has this connectionProperties property. For a mobile
                    # geodatabase network, it will be something like this:
                    # {'dataset': 'main.Streets_ND', 'workspace_factory': 'SQLite', 'connection_info':
                    # {'authentication_mode': 'OSA', 'database': 'main', 'db_connection_properties':
                    # 'C:\\Data\\Packages\\MyPackage\\p14\\sanfrancisco.geodatabase', 'instance':
                    # 'sde:sqlite:C:\\Data\\Packages\\MyPackage\\p14\\sanfrancisco.geodatabase',
                    # 'is_geodatabase': 'true', 'server': 'e:'}}
                    # The 'workspace_factory' property is the relevant one.  For a file geodatabase network, it reads
                    # "File Geodatabase" instead.
                    connection_props = network_obj.connectionProperties
                    is_mgdb = connection_props.get("workspace_factory", None) == "SQLite"
                else:
                    # It's probably a network dataset catalog path
                    network_path = param_network.valueAsText
                    desc = arcpy.Describe(os.path.dirname(os.path.dirname(network_path)))
                    is_mgdb = desc.workspaceFactoryProgID.startswith("esriDataSourcesGDB.SqliteWorkspaceFactory")
                if is_mgdb:
                    param_max_processes.setWarningMessage((
                        "The maximum number of parallel processes is greater than the maximum recommended number "
                        f"({helpers.MAX_RECOMMENDED_MGDB_PROCESSES}) to use with a network dataset in a "
                        "mobile geodatabase."
                    ))
                    return
            except Exception:  # pylint: disable=broad-except
                # If determining the network dataset type fails for some reason, just skip this check
                pass
        # Set a warning if the user has put more processes than the number of logical cores on their machine
        if max_processes > cpu_count():
            param_max_processes.setWarningMessage((
                "The maximum number of parallel processes is greater than the number of logical cores "
                f"({cpu_count()}) in your machine."
            ))


def validate_search_query_param(param_search_query, param_network):
    """Validate the search query parameter for Parallel Calculate Locations."""
    search_query = param_search_query.values
    search_query_sources = [s[0] for s in search_query]

    # Validate source names
    valid_source_names = param_search_query.filters[0].list
    for source in search_query_sources:
        if source not in valid_source_names:
            # Error 030254: Invalid source name: "BadSourceName".
            param_search_query.setIDMessage("Error", 30254, source)
            return

    # Validate no duplicate source names
    if len(set(search_query_sources)) < len(search_query_sources):
        seen = set()
        duplicates = []
        for source in search_query_sources:
            if source in seen:
                duplicates.append(source)
            else:
                seen.add(source)
        # Error 030255: Duplicate source name: "MySourceName".
        param_search_query.setIDMessage("Error", 30255, duplicates[0])
        return

    # Validate SQL queries
    # There's no straightforward way to validate SQL queries in Python, so let's be clever and tricky.  Spin up a dummy
    # SearchCursor using the user's query for each network source as a where clause.  If that throws an exception,
    # return the exception as a validation error.
    # First check that there are any actual queries.  No need to validate if they're all empty.
    populated_queries = [q[1] for q in search_query if q[1]]
    if not populated_queries:
        return
    feature_dataset = os.path.dirname(get_catalog_path(param_network))
    for query in search_query:
        try:
            with arcpy.da.SearchCursor(  # pylint: disable=no-member
                os.path.join(feature_dataset, query[0]), ["OID@"], query[1]
            ) as cur:
                try:
                    next(cur)
                except StopIteration:
                    # The cursor worked but no rows were returned.  This is not an error.
                    pass
        except Exception as ex:  # pylint: disable=broad-except
            param_search_query.setErrorMessage(str(ex))
            return
