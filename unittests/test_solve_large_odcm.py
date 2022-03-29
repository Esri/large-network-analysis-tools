"""Unit tests for the solve_large_odcm.py module.'

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
# pylint: disable=import-error, protected-access, invalid-name

import sys
import os
import datetime
import subprocess
import unittest
from copy import deepcopy
from glob import glob
import arcpy
import portal_credentials  # Contains log-in for an ArcGIS Online account to use as a test portal

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import solve_large_odcm  # noqa: E402, pylint: disable=wrong-import-position
import helpers  # noqa: E402, pylint: disable=wrong-import-position


class TestSolveLargeODCM(unittest.TestCase):
    """Test cases for the solve_large_odcm module."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        self.sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = os.path.join(self.sf_gdb, "Analysis", "TractCentroids")
        self.destinations = os.path.join(self.sf_gdb, "Analysis", "Hospitals")
        self.local_nd = os.path.join(self.sf_gdb, "Transportation", "Streets_ND")
        self.local_tm_time = "Driving Time"
        self.local_tm_dist = "Driving Distance"
        self.portal_nd = portal_credentials.PORTAL_URL
        self.portal_tm = portal_credentials.PORTAL_TRAVEL_MODE
        self.time_of_day_str = "20220329 16:45"

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.PORTAL_USERNAME, portal_credentials.PORTAL_PASSWORD)

        # Create a unique output directory and gdb for this test
        self.scratch_folder = os.path.join(
            CWD, "TestOutput", "Output_SolveLargeODCM_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        # Copy some data to the output gdb to serve as barriers. Do not use tutorial data directly as input because
        # the tests will write network location fields to it, and we don't want to modify the user's original data.
        self.barriers = os.path.join(self.output_gdb, "Barriers")
        arcpy.management.Copy(os.path.join(self.sf_gdb, "Analysis", "CentralDepots"), self.barriers)

        self.od_args = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "output_origins": os.path.join(self.output_gdb, "OutOrigins"),
            "output_destinations": os.path.join(self.output_gdb, "OutDestinations"),
            "chunk_size": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "output_format": "Feature class",
            "output_od_lines": os.path.join(self.output_gdb, "OutODLines"),
            "output_data_folder": None,
            "cutoff": 2,
            "num_destinations": 1,
            "time_of_day": None,
            "precalculate_network_locations": True,
            "barriers": [self.barriers]
        }

    def test_validate_inputs(self):
        """Test the validate_inputs function."""
        invalid_inputs = [
            ("chunk_size", -5, ValueError),
            ("max_processes", 0, ValueError),
            ("cutoff", 0, ValueError),
            ("cutoff", -5, ValueError),
            ("num_destinations", 0, ValueError),
            ("num_destinations", -5, ValueError),
            ("time_units", "BadUnits", ValueError),
            ("distance_units", "BadUnits", ValueError),
            ("origins", os.path.join(self.sf_gdb, "Analysis", "DoesNotExist"), ValueError),
            ("destinations", os.path.join(self.sf_gdb, "Analysis", "DoesNotExist"), ValueError),
            ("barriers", [os.path.join(self.sf_gdb, "Analysis", "DoesNotExist")], ValueError),
            ("network_data_source", os.path.join(self.sf_gdb, "Transportation", "DoesNotExist"), ValueError),
            ("travel_mode", "BadTM", RuntimeError),
            ("time_of_day", "3/29/2022 4:45 PM", ValueError),
            ("time_of_day", "BadDateTime", ValueError)
        ]
        for invalid_input in invalid_inputs:
            property_name = invalid_input[0]
            value = invalid_input[1]
            error_type = invalid_input[2]
            with self.subTest(property_name=property_name, value=value, error_type=error_type):
                inputs = deepcopy(self.od_args)
                inputs[property_name] = value
                od_solver = solve_large_odcm.ODCostMatrixSolver(**inputs)
                with self.assertRaises(error_type):
                    od_solver._validate_inputs()

        # Check validation of missing output feature class or folder location depending on output format
        for output_format in helpers.OUTPUT_FORMATS:
            if output_format == "Feature class":
                output_od_lines = None
                output_data_folder = "Stuff"
            else:
                output_od_lines = "Stuff"
                output_data_folder = None
            with self.subTest(output_format=output_format):
                inputs = deepcopy(self.od_args)
                inputs["output_format"] = output_format
                inputs["output_od_lines"] = output_od_lines
                inputs["output_data_folder"] = output_data_folder
                od_solver = solve_large_odcm.ODCostMatrixSolver(**inputs)
                with self.assertRaises(ValueError):
                    od_solver._validate_inputs()

        # Check validation when the network data source is a service and Arrow output is requested
        # Arrow output from services is not yet supported.
        output_format = "Apache Arrow files"
        with self.subTest(output_format=output_format, network_data_source=self.portal_nd):
            inputs = deepcopy(self.od_args)
            inputs["output_format"] = output_format
            inputs["output_data_folder"] = "Stuff"
            inputs["network_data_source"] = self.portal_nd
            od_solver = solve_large_odcm.ODCostMatrixSolver(**inputs)
            with self.assertRaises(ValueError):
                od_solver._validate_inputs()

    def test_get_tool_limits_and_is_agol(self):
        """Test the _get_tool_limits_and_is_agol function for a portal network data source."""
        inputs = deepcopy(self.od_args)
        inputs["network_data_source"] = self.portal_nd
        inputs["travel_mode"] = self.portal_tm
        od_solver = solve_large_odcm.ODCostMatrixSolver(**inputs)
        od_solver._get_tool_limits_and_is_agol()
        self.assertIsInstance(od_solver.service_limits, dict)
        self.assertIsInstance(od_solver.is_agol, bool)
        self.assertIn("maximumDestinations", od_solver.service_limits)
        self.assertIn("maximumOrigins", od_solver.service_limits)
        if "arcgis.com" in self.portal_nd:
            # Note: If testing with some other portal, this test would need to be updated.
            self.assertTrue(od_solver.is_agol)

    def test_update_max_inputs_for_service(self):
        """Test the update_max_inputs_for_service function."""
        max_origins = 1500
        max_destinations = 700

        inputs = deepcopy(self.od_args)
        inputs["network_data_source"] = self.portal_nd
        inputs["travel_mode"] = self.portal_tm
        od_solver = solve_large_odcm.ODCostMatrixSolver(**inputs)
        od_solver.max_origins = max_origins
        od_solver.max_destinations = max_destinations
        tool_limits = {
            'forceHierarchyBeyondDistance': 50.0,
            'forceHierarchyBeyondDistanceUnits':
            'Miles',
            'maximumDestinations': 1000.0,
            'maximumFeaturesAffectedByLineBarriers': 500.0,
            'maximumFeaturesAffectedByPointBarriers': 250.0,
            'maximumFeaturesAffectedByPolygonBarriers': 2000.0,
            'maximumGeodesicDistanceUnitsWhenWalking': 'Miles',
            'maximumGeodesicDistanceWhenWalking': 27.0,
            'maximumOrigins': 1000.0
        }
        od_solver.service_limits = tool_limits
        od_solver._update_max_inputs_for_service()
        self.assertEqual(tool_limits["maximumOrigins"], od_solver.max_origins)
        self.assertEqual(max_destinations, od_solver.max_destinations)

        # Test when there are no limits
        tool_limits = {
            'forceHierarchyBeyondDistance': 50.0,
            'forceHierarchyBeyondDistanceUnits':
            'Miles',
            'maximumDestinations': None,
            'maximumFeaturesAffectedByLineBarriers': 500.0,
            'maximumFeaturesAffectedByPointBarriers': 250.0,
            'maximumFeaturesAffectedByPolygonBarriers': 2000.0,
            'maximumGeodesicDistanceUnitsWhenWalking': 'Miles',
            'maximumGeodesicDistanceWhenWalking': 27.0,
            'maximumOrigins': None
        }
        od_solver.max_origins = max_origins
        od_solver.max_destinations = max_destinations
        od_solver.service_limits = tool_limits
        od_solver._update_max_inputs_for_service()
        self.assertEqual(max_origins, od_solver.max_origins)
        self.assertEqual(max_destinations, od_solver.max_destinations)

    def test_spatially_sort_input(self):
        """Test the spatially_sort_input function."""
        od_solver = solve_large_odcm.ODCostMatrixSolver(**self.od_args)
        fc_to_sort = os.path.join(self.output_gdb, "Sorted")
        arcpy.management.Copy(self.destinations, fc_to_sort)
        od_solver._spatially_sort_input(fc_to_sort, "DestinationOID")
        self.assertTrue(arcpy.Exists(fc_to_sort))
        self.assertIn("DestinationOID", [f.name for f in arcpy.ListFields(fc_to_sort)])

    def test_precalculate_network_locations(self):
        """Test the precalculate_network_locations function."""
        loc_fields = set(["SourceID", "SourceOID", "PosAlong", "SideOfEdge"])

        # Precalculate network locations
        od_solver = solve_large_odcm.ODCostMatrixSolver(**self.od_args)
        fc_to_precalculate = os.path.join(self.output_gdb, "Precalculated")
        arcpy.management.Copy(self.destinations, fc_to_precalculate)
        od_solver._precalculate_network_locations(fc_to_precalculate)
        actual_fields = set([f.name for f in arcpy.ListFields(fc_to_precalculate)])
        self.assertTrue(loc_fields.issubset(actual_fields), "Network location fields not added")
        for row in arcpy.da.SearchCursor(fc_to_precalculate, list(loc_fields)):  # pylint: disable=no-member
            for val in row:
                self.assertIsNotNone(val)

    def test_solve_large_od_cost_matrix_featureclass(self):
        """Test the full solve OD Cost Matrix workflow with feature class output."""
        od_solver = solve_large_odcm.ODCostMatrixSolver(**self.od_args)
        od_solver.solve_large_od_cost_matrix()
        self.assertTrue(arcpy.Exists(self.od_args["output_od_lines"]))
        self.assertTrue(arcpy.Exists(self.od_args["output_origins"]))
        self.assertTrue(arcpy.Exists(self.od_args["output_destinations"]))

    def test_solve_large_od_cost_matrix_same_inputs_csv(self):
        """Test the full solve OD Cost Matrix workflow when origins and destinations are the same. Use CSV outputs."""
        out_folder = os.path.join(self.scratch_folder, "FullWorkflow_CSV_SameInputs")
        os.mkdir(out_folder)
        od_args = {
            "origins": self.destinations,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "output_origins": os.path.join(self.output_gdb, "OutOriginsSame"),
            "output_destinations": os.path.join(self.output_gdb, "OutDestinationsSame"),
            "chunk_size": 50,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "output_format": "CSV files",
            "output_od_lines": None,
            "output_data_folder": out_folder,
            "cutoff": 2,
            "num_destinations": 2,
            "time_of_day": self.time_of_day_str,
            "precalculate_network_locations": True,
            "barriers": ""
        }
        od_solver = solve_large_odcm.ODCostMatrixSolver(**od_args)
        od_solver.solve_large_od_cost_matrix()
        self.assertTrue(arcpy.Exists(od_args["output_origins"]))
        self.assertTrue(arcpy.Exists(od_args["output_destinations"]))
        csv_files = glob(os.path.join(out_folder, "*.csv"))
        self.assertGreater(len(csv_files), 0)

    def test_solve_large_od_cost_matrix_arrow(self):
        """Test the full solve OD Cost Matrix workflow with Arrow table outputs"""
        out_folder = os.path.join(self.scratch_folder, "FullWorkflow_Arrow")
        os.mkdir(out_folder)
        od_args = {
            "origins": self.destinations,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "output_origins": os.path.join(self.output_gdb, "OutOriginsArrow"),
            "output_destinations": os.path.join(self.output_gdb, "OutDestinationsArrow"),
            "chunk_size": 50,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "output_format": "Apache Arrow files",
            "output_od_lines": None,
            "output_data_folder": out_folder,
            "cutoff": 2,
            "num_destinations": 2,
            "time_of_day": None,
            "precalculate_network_locations": True,
            "barriers": ""
        }
        od_solver = solve_large_odcm.ODCostMatrixSolver(**od_args)
        od_solver.solve_large_od_cost_matrix()
        self.assertTrue(arcpy.Exists(od_args["output_origins"]))
        self.assertTrue(arcpy.Exists(od_args["output_destinations"]))
        arrow_files = glob(os.path.join(out_folder, "*.arrow"))
        self.assertGreater(len(arrow_files), 0)

    def test_cli(self):
        """Test the command line interface of solve_large_odcm."""
        out_folder = os.path.join(self.scratch_folder, "CLI_CSV_Output")
        os.mkdir(out_folder)
        odcm_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(os.path.dirname(CWD), "solve_large_odcm.py"),
            "--origins", self.origins,
            "--destinations", self.destinations,
            "--output-origins", os.path.join(self.output_gdb, "OutOriginsCLI"),
            "--output-destinations", os.path.join(self.output_gdb, "OutDestinationsCLI"),
            "--network-data-source", self.local_nd,
            "--travel-mode", self.local_tm_time,
            "--time-units", "Minutes",
            "--distance-units", "Miles",
            "--output-format", "CSV files",
            "--output-data-folder", out_folder,
            "--chunk-size", "50",
            "--max-processes", "4",
            "--cutoff", "10",
            "--num-destinations", "1",
            "--time-of-day", self.time_of_day_str,
            "--precalculate-network-locations", "true",
            "--barriers", self.barriers
        ]
        result = subprocess.run(odcm_inputs)
        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
