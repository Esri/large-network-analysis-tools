"""Unit tests for the solve_large_route_pair_analysis.py module.

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
import arcpy
import portal_credentials  # Contains log-in for an ArcGIS Online account to use as a test portal
import input_data_helper

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import solve_large_route_pair_analysis  # noqa: E402, pylint: disable=wrong-import-position
import helpers  # noqa: E402, pylint: disable=wrong-import-position


class TestSolveLargeRoutePairAnalysis(unittest.TestCase):
    """Test cases for the solve_large_route_pair_analysis module."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        self.sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = input_data_helper.get_tract_centroids_with_store_id_fc(sf_gdb)
        self.destinations = os.path.join(self.sf_gdb, "Analysis", "Stores")
        self.local_nd = os.path.join(self.sf_gdb, "Transportation", "Streets_ND")
        self.local_tm_time = "Driving Time"
        self.local_tm_dist = "Driving Distance"
        self.portal_nd = portal_credentials.PORTAL_URL
        self.portal_tm = portal_credentials.PORTAL_TRAVEL_MODE
        self.time_of_day_str = "20220329 16:45"

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.PORTAL_USERNAME, portal_credentials.PORTAL_PASSWORD)

        # Create a unique output directory and gdb for this test
        self.scratch_folder = os.path.join(
            CWD, "TestOutput", "Output_SolveLargeRoutePair_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        # Copy some data to the output gdb to serve as barriers. Do not use tutorial data directly as input because
        # the tests will write network location fields to it, and we don't want to modify the user's original data.
        self.barriers = os.path.join(self.output_gdb, "Barriers")
        arcpy.management.Copy(os.path.join(self.sf_gdb, "Analysis", "CentralDepots"), self.barriers)

        self.rt_args = {
            "origins": self.origins,
            "origin_id_field": "ID",
            "assigned_dest_field": "StoreID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "chunk_size": 15,
            "max_processes": 4,
            "output_routes": os.path.join(self.output_gdb, "OutRoutes"),
            "time_of_day": self.time_of_day_str,
            "barriers": "",
            "precalculate_network_locations": True,
            "sort_origins": True
        }

    def test_validate_inputs(self):
        """Test the validate_inputs function."""
        does_not_exist = os.path.join(self.sf_gdb, "Analysis", "DoesNotExist")
        invalid_inputs = [
            ("chunk_size", -5, ValueError, "Chunk size must be greater than 0."),
            ("max_processes", 0, ValueError, "Maximum allowed parallel processes must be greater than 0."),
            ("time_units", "BadUnits", ValueError, "Invalid time units: BadUnits"),
            ("distance_units", "BadUnits", ValueError, "Invalid distance units: BadUnits"),
            ("origins", does_not_exist, ValueError, f"Input dataset {does_not_exist} does not exist."),
            ("destinations", does_not_exist, ValueError, f"Input dataset {does_not_exist} does not exist."),
            ("barriers", [does_not_exist], ValueError, f"Input dataset {does_not_exist} does not exist."),
            ("network_data_source", does_not_exist, ValueError,
             f"Input network dataset {does_not_exist} does not exist."),
            ("travel_mode", "BadTM", RuntimeError, ""),
            ("time_of_day", "3/29/2022 4:45 PM", ValueError, ""),
            ("time_of_day", "BadDateTime", ValueError, ""),
            ("origin_id_field", "BadField", ValueError,
             f"Unique ID field BadField does not exist in dataset {self.origins}."),
            ("origin_id_field", "STATE_NAME", ValueError,
             f"Non-unique values were found in the unique ID field STATE_NAME in {self.origins}."),
            ("dest_id_field", "BadField", ValueError,
             f"Unique ID field BadField does not exist in dataset {self.destinations}."),
            ("dest_id_field", "ServiceTime", ValueError,
             f"Non-unique values were found in the unique ID field ServiceTime in {self.destinations}."),
            ("assigned_dest_field", "BadField", ValueError,
             f"Assigned destination field BadField does not exist in Origins dataset {self.origins}."),
            ("assigned_dest_field", "STATE_NAME", ValueError,
             (f"All origins in the Origins dataset {self.origins} have invalid values in the assigned "
              f"destination field STATE_NAME that do not correspond to values in the "
              f"destinations unique ID field NAME in {self.destinations}. Ensure that you "
              "have chosen the correct datasets and fields and that the field types match."))
        ]
        for invalid_input in invalid_inputs:
            property_name, value, error_type, expected_message = invalid_input
            with self.subTest(
                property_name=property_name, value=value, error_type=error_type, expected_message=expected_message
            ):
                inputs = deepcopy(self.rt_args)
                inputs[property_name] = value
                rt_solver = solve_large_route_pair_analysis.RoutePairSolver(**inputs)
                with self.assertRaises(error_type) as ex:
                    rt_solver._validate_inputs()
                if expected_message:
                    self.assertEqual(expected_message, str(ex.exception))

    def test_update_max_inputs_for_service(self):
        """Test the update_max_inputs_for_service function."""
        max_routes = 20000000
        inputs = deepcopy(self.rt_args)
        inputs["network_data_source"] = self.portal_nd
        inputs["travel_mode"] = self.portal_tm
        inputs["chunk_size"] = max_routes
        rt_solver = solve_large_route_pair_analysis.RoutePairSolver(**inputs)
        tool_limits = {
            'forceHierarchyBeyondDistance': 50.0,
            'forceHierarchyBeyondDistanceUnits': 'Miles',
            'maximumFeaturesAffectedByLineBarriers': 500.0,
            'maximumFeaturesAffectedByPointBarriers': 250.0,
            'maximumFeaturesAffectedByPolygonBarriers': 2000.0,
            'maximumGeodesicDistanceUnitsWhenWalking': 'Miles',
            'maximumGeodesicDistanceWhenWalking': 27.0,
            'maximumStops': 10000.0,
            'maximumStopsPerRoute': 150.0
        }
        rt_solver.service_limits = tool_limits
        rt_solver._update_max_inputs_for_service()
        self.assertEqual(tool_limits["maximumStops"] / 2, rt_solver.chunk_size)

        # Test when there are no limits
        tool_limits = {
            'forceHierarchyBeyondDistance': 50.0,
            'forceHierarchyBeyondDistanceUnits': 'Miles',
            'maximumFeaturesAffectedByLineBarriers': 500.0,
            'maximumFeaturesAffectedByPointBarriers': 250.0,
            'maximumFeaturesAffectedByPolygonBarriers': 2000.0,
            'maximumGeodesicDistanceUnitsWhenWalking': 'Miles',
            'maximumGeodesicDistanceWhenWalking': 27.0,
            'maximumStops': None,
            'maximumStopsPerRoute': 150.0
        }
        rt_solver.chunk_size = max_routes
        rt_solver.service_limits = tool_limits
        rt_solver._update_max_inputs_for_service()
        self.assertEqual(max_routes, rt_solver.chunk_size)

    def test_solve_large_route_pair_analysis(self):
        """Test the full solve route pair workflow."""
        rt_solver = solve_large_route_pair_analysis.RoutePairSolver(**self.rt_args)
        rt_solver.solve_large_route_pair_analysis()
        self.assertTrue(arcpy.Exists(self.rt_args["output_routes"]))

    def test_cli(self):
        """Test the command line interface of solve_large_route_pair_analysis."""
        out_folder = os.path.join(self.scratch_folder, "CLI_CSV_Output")
        os.mkdir(out_folder)
        rt_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(os.path.dirname(CWD), "solve_large_route_pair_analysis.py"),
            "--origins", self.origins,
            "--origins-id-field", "ID",
            "--assigned-dest-field", "StoreID",
            "--destinations", self.destinations,
            "--destinations-id-field", "NAME",
            "--network-data-source", self.local_nd,
            "--travel-mode", self.local_tm_dist,
            "--time-units", "Minutes",
            "--distance-units", "Miles",
            "--max-routes", "15",
            "--max-processes", "4",
            "--out-routes", os.path.join(self.output_gdb, "OutCLIRoutes"),
            "--time-of-day", self.time_of_day_str,
            "--precalculate-network-locations", "true",
            "--sort-origins", "true"
        ]
        result = subprocess.run(rt_inputs, check=True)
        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
