"""Unit tests for the parallel_route_pairs.py module.

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
import portal_credentials
import input_data_helper

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import parallel_route_pairs  # noqa: E402, pylint: disable=wrong-import-position


class TestParallelRoutePairs(unittest.TestCase):
    """Test cases for the parallel_route_pairs module."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        """Set up shared test properties."""
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = input_data_helper.get_tract_centroids_with_store_id_fc(sf_gdb)
        self.destinations = os.path.join(sf_gdb, "Analysis", "Stores")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        self.local_tm_time = "Driving Time"
        self.local_tm_dist = "Driving Distance"
        self.portal_nd = portal_credentials.PORTAL_URL
        self.portal_tm = portal_credentials.PORTAL_TRAVEL_MODE

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.PORTAL_USERNAME, portal_credentials.PORTAL_PASSWORD)

        # Create a unique output directory and gdb for this test
        self.output_folder = os.path.join(
            CWD, "TestOutput", "Output_ParallelRoutePairs_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.output_folder)
        self.output_gdb = os.path.join(self.output_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        self.route_args = {
            "origins": self.origins,
            "origin_id_field": "ID",
            "assigned_dest_field": "StoreID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "time_of_day": None,
            "reverse_direction": False,
            "scratch_folder": self.output_folder,
            "origin_transfer_fields": [],
            "destination_transfer_fields": [],
            "barriers": []
        }

        self.parallel_rt_class_args = {
            "origins": self.origins,
            "origin_id_field": "ID",
            "assigned_dest_field": "StoreID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "max_routes": 15,
            "max_processes": 4,
            "out_routes": os.path.join(self.output_gdb, "OutRoutes"),
            "reverse_direction": False,
            "scratch_folder": self.output_folder,  # Should be set within test if real output will be written
            "time_of_day": "20220329 16:45",
            "barriers": ""
        }

    def test_Route_select_inputs(self):
        """Test the _select_inputs method of the Route class."""
        rt = parallel_route_pairs.Route(**self.route_args)
        origin_criteria = [4, 9]  # Encompasses 6 rows
        rt._select_inputs(origin_criteria)
        self.assertEqual(6, int(arcpy.management.GetCount(rt.input_origins_layer_obj).getOutput(0)))

    def test_Route_solve(self):
        """Test the solve method of the Route class with feature class output."""
        # Initialize an Route analysis object
        rt = parallel_route_pairs.Route(**self.route_args)
        # Solve a chunk
        origin_criteria = [2, 12]  # 11 rows
        rt.solve(origin_criteria)
        # Check results
        self.assertIsInstance(rt.job_result, dict)
        self.assertTrue(rt.job_result["solveSucceeded"], "Route solve failed")
        self.assertTrue(arcpy.Exists(rt.job_result["outputRoutes"]), "Route output does not exist.")
        # Expect 9 rows because two of the StoreID values are bad and are skipped
        self.assertEqual(9, int(arcpy.management.GetCount(rt.job_result["outputRoutes"]).getOutput(0)))
        # Make sure the ID fields have been added
        route_fields = [f.name for f in arcpy.ListFields(rt.job_result["outputRoutes"])]
        self.assertIn("OriginUniqueID", route_fields, "Routes output missing OriginUniqueID field.")
        self.assertIn("DestinationUniqueID", route_fields, "Routes output missing DestinationUniqueID field.")

    def test_ParallelRoutePairCalculator_validate_route_settings(self):
        """Test the _validate_route_settings function."""
        # Test that with good inputs, we return the correct optimized field name
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**self.parallel_rt_class_args)
        rt_calculator._validate_route_settings()
        # Test completely invalid travel mode
        rt_inputs = deepcopy(self.parallel_rt_class_args)
        rt_inputs["travel_mode"] = "InvalidTM"
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**rt_inputs)
        with self.assertRaises(RuntimeError):
            rt_calculator._validate_route_settings()

    def test_ParallelRoutePairCalculator_solve_route_in_parallel(self):
        """Test the solve_od_in_parallel function. Output to feature class."""
        out_routes = os.path.join(self.output_gdb, "Out_Combined_Routes")
        scratch_folder = os.path.join(self.output_folder, "Out_Combined_Routes")
        os.mkdir(scratch_folder)
        rt_inputs = deepcopy(self.parallel_rt_class_args)
        rt_inputs["out_routes"] = out_routes
        rt_inputs["scratch_folder"] = scratch_folder

        # Run parallel process. This calculates the OD and also post-processes the results
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**rt_inputs)
        rt_calculator.solve_route_in_parallel()

        # Check results
        self.assertTrue(arcpy.Exists(out_routes))
        # There are 208 tract centroids, but three of them have null or invalid assigned destinations
        self.assertEqual(205, int(arcpy.management.GetCount(out_routes).getOutput(0)))

    def test_cli(self):
        """Test the command line interface of and launch_parallel_rt_pairs function."""
        out_folder = os.path.join(self.output_folder, "CLI_Output")
        os.mkdir(out_folder)
        rt_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(os.path.dirname(CWD), "parallel_route_pairs.py"),
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
            "--reverse-direction", "false",
            "--scratch-folder", out_folder,
            "--time-of-day", "20220329 16:45"
        ]
        result = subprocess.run(rt_inputs, check=True)
        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
