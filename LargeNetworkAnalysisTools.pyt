"""Python toolbox that defines tools for solving large network analysis problems.

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
# Disable a bunch of linter errors caused by the standard python toolbox class definitions that we cannot change
# pylint: disable=invalid-name, useless-object-inheritance, too-few-public-methods, no-self-use, too-many-locals
# pylint: disable=useless-return, unused-argument
import arcpy

import helpers
from solve_large_odcm import ODCostMatrixSolver


class Toolbox(object):
    """Tools for solving large network analysis problems."""

    def __init__(self):
        """Define the toolbox."""
        self.label = "Large Network Analysis Tools"
        self.alias = "LargeNetworkAnalysisTools"

        # List of tool classes associated with this toolbox
        self.tools = [SolveLargeODCostMatrix]


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

        param_out_od_lines = arcpy.Parameter(
            displayName="Output OD Lines Feature Class",
            name="Output_OD_Lines_Feature_Class",
            datatype="DEFeatureClass",
            parameterType="Required",
            direction="Output"
        )

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

        params = [
            param_origins,
            param_destinations,
            param_out_od_lines,
            param_out_origins,
            param_out_destinations,
            param_network,
            param_travel_mode,
            param_time_units,
            param_distance_units,
            param_chunk_size,
            param_max_processes,
            param_cutoff,
            param_num_dests,
            param_barriers,
            param_precalculate_network_locations
        ]

        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        param_network = parameters[5]
        param_precalculate = parameters[14]

        # Turn off and hide Precalculate Network Locations parameter if the network data source is a service
        if param_network.altered and param_network.value:
            if helpers.is_nds_service(param_network.valueAsText):
                param_precalculate.value = False
                param_precalculate.enabled = False
            else:
                param_precalculate.enabled = True

        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        param_network = parameters[5]
        param_max_processes = parameters[10]

        # If the network data source is arcgis.com, cap max processes
        if param_max_processes.altered and param_max_processes.value and \
                param_network.altered and param_network.value:
            if "arcgis.com" in param_network.valueAsText and param_max_processes.value > helpers.MAX_AGOL_PROCESSES:
                param_max_processes.setErrorMessage((
                    f"The maximum number of parallel processes cannot exceed {helpers.MAX_AGOL_PROCESSES} when the "
                    "ArcGIS Online services are used as the network data source."
                ))

        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # Initialize the solver class
        od_solver = ODCostMatrixSolver(
            parameters[0].value,  # origins
            parameters[1].value,  # destinations
            get_catalog_path(parameters[5]),  # network
            parameters[6].value,  # travel mode
            parameters[2].valueAsText,  # output OD lines
            parameters[3].valueAsText,  # output origins
            parameters[4].valueAsText,  # output destinations
            parameters[9].value,  # chunk size
            parameters[10].value,  # max processes
            parameters[7].valueAsText,  # time units
            parameters[8].valueAsText,  # distance units
            parameters[11].valueAsText,  # cutoff
            parameters[12].valueAsText,  # number of destinations to find
            parameters[14].value,  # Should precalculate network locations
            get_catalog_path_multivalue(parameters[13])  # barriers
        )

        # Solve the OD Cost Matrix analysis
        od_solver.solve_large_od_cost_matrix()

        return


def get_catalog_path(param):
    """Get the catalog path for a single value parameter if possible.

    Args:
        param (arcpy.Parameter): Parameter from which to retrieve the catalog path.

    Returns:
        list(str): List of catalog paths to the data
    """
    param_value = param.value
    if not param_value:
        return ""
    # If the value is a layer object, get its data source (catalog path)
    if hasattr(param_value, "dataSource"):
        return param_value.dataSource
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
            catalog_paths.append(val.dataSource)
        # Otherwise, it's probably already a string catalog path. The only way to get it is to retrive it from the
        # valueAsText string that we split up above.
        else:
            catalog_paths.append(string_values[idx])
    return catalog_paths
