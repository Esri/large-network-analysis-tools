"""Unit tests for the parallel_route_pairs.py module.

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
# pylint: disable=import-error, protected-access, invalid-name

import sys
import os
import datetime
import subprocess
import pandas as pd
import unittest
from copy import deepcopy
import arcpy
import portal_credentials
import input_data_helper

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import parallel_route_pairs  # noqa: E402, pylint: disable=wrong-import-position
from helpers import arcgis_version, PreassignedODPairType  # noqa: E402, pylint: disable=wrong-import-position


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
        self.od_pair_table = input_data_helper.get_od_pair_csv(self.input_data_folder)
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

        self.route_args_one_to_one = {
            "pair_type": PreassignedODPairType.one_to_one,
            "origins": self.origins,
            "origin_id_field": "ID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "time_of_day": None,
            "reverse_direction": False,
            "scratch_folder": self.output_folder,
            "assigned_dest_field": "StoreID",
            "od_pair_table": None,
            "origin_transfer_fields": [],
            "destination_transfer_fields": [],
            "barriers": []
        }
        self.route_args_many_to_many = {
            "pair_type": PreassignedODPairType.many_to_many,
            "origins": self.origins,
            "origin_id_field": "ID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "time_of_day": None,
            "reverse_direction": False,
            "scratch_folder": self.output_folder,
            "assigned_dest_field": None,
            "od_pair_table": self.od_pair_table,
            "origin_transfer_fields": [],
            "destination_transfer_fields": [],
            "barriers": []
        }
        self.parallel_rt_class_args_one_to_one = {
            "pair_type_str": "one_to_one",
            "origins": self.origins,
            "origin_id_field": "ID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "max_routes": 15,
            "max_processes": 4,
            "out_routes": os.path.join(self.output_gdb, "OutRoutes_OneToOne"),
            "reverse_direction": False,
            "scratch_folder": self.output_folder,  # Should be set within test if real output will be written
            "assigned_dest_field": "StoreID",
            "od_pair_table": None,
            "time_of_day": "20220329 16:45",
            "barriers": ""
        }
        self.parallel_rt_class_args_many_to_many = {
            "pair_type_str": "many_to_many",
            "origins": self.origins,
            "origin_id_field": "ID",
            "destinations": self.destinations,
            "dest_id_field": "NAME",
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "max_routes": 15,
            "max_processes": 4,
            "out_routes": os.path.join(self.output_gdb, "OutRoutes_ManyToMany"),
            "reverse_direction": False,
            "scratch_folder": self.output_folder,  # Should be set within test if real output will be written
            "assigned_dest_field": None,
            "od_pair_table": self.od_pair_table,
            "time_of_day": "20220329 16:45",
            "barriers": ""
        }

    def test_Route_get_od_pairs_for_chunk(self):
        """Test the _get_od_pairs_for_chunk method of the Route class."""
        rt = parallel_route_pairs.Route(**self.route_args_many_to_many)
        chunk_size = 10
        chunk_num = 3
        # Get the third chunk in the OD pairs file
        chunk_definition = [chunk_num, chunk_size]
        rt._get_od_pairs_for_chunk(chunk_definition)
        # Make sure we have the right number of OD pairs
        self.assertEqual(chunk_size, len(rt.od_pairs))
        # Verify that the solver's OD pairs are the right ones.
        df_od_pairs = pd.read_csv(
            self.od_pair_table,
            header=None,
            dtype=str
        )
        chunk_start = chunk_num * chunk_size
        expected_od_pairs = df_od_pairs.loc[chunk_start:chunk_start + chunk_size - 1].values.tolist()
        self.assertEqual(expected_od_pairs, rt.od_pairs)

    def test_Route_select_inputs_one_to_one(self):
        """Test the _select_inputs_one_to_one method of the Route class."""
        rt = parallel_route_pairs.Route(**self.route_args_one_to_one)
        origin_criteria = [4, 9]  # Encompasses 6 rows
        rt._select_inputs_one_to_one(origin_criteria)
        self.assertEqual(6, int(arcpy.management.GetCount(rt.input_origins_layer_obj).getOutput(0)))

    def test_Route_select_inputs_many_to_many_str(self):
        """Test the _select_inputs_many_to_many method of the Route class using string-type ID fields."""
        rt = parallel_route_pairs.Route(**self.route_args_many_to_many)
        rt.od_pairs = [  # 3 unique origins, 5 unique destinations
            ["06075060700", "Store_19"],
            ["06081601400", "Store_7"],
            ["06081601400", "Store_15"],
            ["06081601400", "Store_2"],
            ["06075023001", "Store_15"],
            ["06075023001", "Store_25"]
        ]
        rt._select_inputs_many_to_many()
        self.assertEqual(3, int(arcpy.management.GetCount(rt.input_origins_layer_obj).getOutput(0)))
        self.assertEqual(5, int(arcpy.management.GetCount(rt.input_dests_layer_obj).getOutput(0)))

    def test_Route_select_inputs_many_to_many_num(self):
        """Test the _select_inputs_many_to_many method of the Route class using numerical ID fields."""
        rt = parallel_route_pairs.Route(**self.route_args_many_to_many)
        rt.origin_id_field = "ObjectID"
        rt.dest_id_field = "ObjectID"
        rt.od_pairs = [  # 3 unique origins, 5 unique destinations
            [2, 16],
            [4, 19],
            [4, 5],
            [4, 10],
            [7, 5],
            [7, 25]
        ]
        rt._select_inputs_many_to_many()
        self.assertEqual(3, int(arcpy.management.GetCount(rt.input_origins_layer_obj).getOutput(0)))
        self.assertEqual(5, int(arcpy.management.GetCount(rt.input_dests_layer_obj).getOutput(0)))

    def test_Route_solve_one_to_one(self):
        """Test the solve method of the Route class with feature class output for the one-to-one pair type."""
        # Initialize an Route analysis object
        rt = parallel_route_pairs.Route(**self.route_args_one_to_one)
        # Solve a chunk
        origin_criteria = [2, 12]  # 11 rows
        rt.solve(origin_criteria)
        # Check results
        self.assertIsInstance(rt.job_result, dict)
        self.assertTrue(rt.job_result["solveSucceeded"], "Route solve failed")
        self.assertTrue(arcpy.Exists(rt.job_result["outputRoutes"]), "Route output does not exist.")
        # Expect 9 rows because two of the StoreID values are bad and are skipped
        self.assertEqual(9, int(arcpy.management.GetCount(rt.job_result["outputRoutes"]).getOutput(0)))
        # Make sure the ID fields have been added and populated
        route_fields = [f.name for f in arcpy.ListFields(rt.job_result["outputRoutes"])]
        self.assertIn("OriginUniqueID", route_fields, "Routes output missing OriginUniqueID field.")
        self.assertIn("DestinationUniqueID", route_fields, "Routes output missing DestinationUniqueID field.")
        for row in arcpy.da.SearchCursor(rt.job_result["outputRoutes"], ["OriginUniqueID", "DestinationUniqueID"]):
            self.assertIsNotNone(row[0], "Null OriginUniqueID field value in output routes.")
            self.assertIsNotNone(row[1], "Null DestinationUniqueID field value in output routes.")

    def test_Route_solve_many_to_many(self):
        """Test the solve method of the Route class with feature class output for the many-to-many pair type."""
        # Initialize an Route analysis object
        rt = parallel_route_pairs.Route(**self.route_args_many_to_many)
        # Solve a chunk
        chunk_size = 10
        chunk_num = 2  # Corresponds to OD pairs 2-20
        chunk_definition = [chunk_num, chunk_size]
        rt.solve(chunk_definition)
        # Check results
        self.assertIsInstance(rt.job_result, dict)
        self.assertTrue(rt.job_result["solveSucceeded"], "Route solve failed")
        self.assertTrue(arcpy.Exists(rt.job_result["outputRoutes"]), "Route output does not exist.")
        # Check for correct number of routes
        self.assertEqual(chunk_size, int(arcpy.management.GetCount(rt.job_result["outputRoutes"]).getOutput(0)))
        # Make sure the ID fields have been added and populated
        route_fields = [f.name for f in arcpy.ListFields(rt.job_result["outputRoutes"])]
        self.assertIn("OriginUniqueID", route_fields, "Routes output missing OriginUniqueID field.")
        self.assertIn("DestinationUniqueID", route_fields, "Routes output missing DestinationUniqueID field.")
        df_od_pairs = pd.read_csv(
            self.od_pair_table,
            header=None,
            skiprows=chunk_size*chunk_num,
            nrows=chunk_size,
            dtype=str
        )
        expected_origin_ids = df_od_pairs[0].unique().tolist()
        expected_dest_ids = df_od_pairs[1].unique().tolist()
        for row in arcpy.da.SearchCursor(rt.job_result["outputRoutes"], ["OriginUniqueID", "DestinationUniqueID"]):
            self.assertIsNotNone(row[0], "Null OriginUniqueID field value in output routes.")
            self.assertIn(row[0], expected_origin_ids, "OriginUniqueID does not match expected list of IDs.")
            self.assertIsNotNone(row[1], "Null DestinationUniqueID field value in output routes.")
            self.assertIn(row[1], expected_dest_ids, "DestinationUniqueID does not match expected list of IDs.")

    def test_Route_solve_service_one_to_one(self):
        """Test the solve method of the Route class with feature class output using a service."""
        # Initialize an Route analysis object
        route_args = deepcopy(self.route_args_one_to_one)
        route_args["network_data_source"] = self.portal_nd
        route_args["travel_mode"] = self.portal_tm
        rt = parallel_route_pairs.Route(**route_args)
        # Solve a chunk
        origin_criteria = [2, 12]  # 11 rows
        rt.solve(origin_criteria)
        # Check results
        self.assertIsInstance(rt.job_result, dict)
        self.assertTrue(rt.job_result["solveSucceeded"], "Route solve failed")
        self.assertTrue(arcpy.Exists(rt.job_result["outputRoutes"]), "Route output does not exist.")
        # Expect 9 rows because two of the StoreID values are bad and are skipped
        self.assertEqual(9, int(arcpy.management.GetCount(rt.job_result["outputRoutes"]).getOutput(0)))
        # Make sure the ID fields have been added and populated
        route_fields = [f.name for f in arcpy.ListFields(rt.job_result["outputRoutes"])]
        self.assertIn("OriginUniqueID", route_fields, "Routes output missing OriginUniqueID field.")
        self.assertIn("DestinationUniqueID", route_fields, "Routes output missing DestinationUniqueID field.")
        for row in arcpy.da.SearchCursor(rt.job_result["outputRoutes"], ["OriginUniqueID", "DestinationUniqueID"]):
            self.assertIsNotNone(row[0], "Null OriginUniqueID field value in output routes.")
            self.assertIsNotNone(row[1], "Null DestinationUniqueID field value in output routes.")

    def test_ParallelRoutePairCalculator_validate_route_settings(self):
        """Test the _validate_route_settings function."""
        # Test that with good inputs, we return the correct optimized field name
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**self.parallel_rt_class_args_one_to_one)
        rt_calculator._validate_route_settings()
        # Test completely invalid travel mode
        rt_inputs = deepcopy(self.parallel_rt_class_args_one_to_one)
        rt_inputs["travel_mode"] = "InvalidTM"
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**rt_inputs)
        error_type = ValueError if arcgis_version >= "3.1" else RuntimeError
        with self.assertRaises(error_type):
            rt_calculator._validate_route_settings()

    def test_ParallelRoutePairCalculator_solve_route_in_parallel_one_to_one(self):
        """Test the solve_od_in_parallel function with the one-to-one pair type. Output to feature class."""
        out_routes = os.path.join(self.output_gdb, "Out_Combined_Routes_OneToOne")
        scratch_folder = os.path.join(self.output_folder, "Out_Combined_Routes_OneToOne")
        os.mkdir(scratch_folder)
        rt_inputs = deepcopy(self.parallel_rt_class_args_one_to_one)
        rt_inputs["out_routes"] = out_routes
        rt_inputs["scratch_folder"] = scratch_folder

        # Run parallel process. This calculates the OD and also post-processes the results
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**rt_inputs)
        rt_calculator.solve_route_in_parallel()

        # Check results
        self.assertTrue(arcpy.Exists(out_routes))
        # There are 208 tract centroids, but three of them have null or invalid assigned destinations
        self.assertEqual(205, int(arcpy.management.GetCount(out_routes).getOutput(0)))

    def test_ParallelRoutePairCalculator_solve_route_in_parallel_many_to_many(self):
        """Test the solve_od_in_parallel function with the many-to-many pair type. Output to feature class."""
        out_routes = os.path.join(self.output_gdb, "Out_Combined_Routes_ManyToMany")
        scratch_folder = os.path.join(self.output_folder, "Out_Combined_Routes_ManyToMany")
        os.mkdir(scratch_folder)
        rt_inputs = deepcopy(self.parallel_rt_class_args_many_to_many)
        rt_inputs["out_routes"] = out_routes
        rt_inputs["scratch_folder"] = scratch_folder

        # Run parallel process. This calculates the OD and also post-processes the results
        rt_calculator = parallel_route_pairs.ParallelRoutePairCalculator(**rt_inputs)
        rt_calculator.solve_route_in_parallel()

        # Check results
        self.assertTrue(arcpy.Exists(out_routes))
        # The CSV file has 63 lines, so we should get 63 routes.
        self.assertEqual(63, int(arcpy.management.GetCount(out_routes).getOutput(0)))

    def test_cli_one_to_one(self):
        """Test the command line interface of and launch_parallel_rt_pairs function for the one-to-one pair type."""
        out_folder = os.path.join(self.output_folder, "CLI_Output_OneToOne")
        os.mkdir(out_folder)
        out_routes = os.path.join(self.output_gdb, "OutCLIRoutes_OneToOne")
        rt_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(os.path.dirname(CWD), "parallel_route_pairs.py"),
            "--pair-type", "one_to_one",
            "--origins", self.origins,
            "--origins-id-field", "ID",
            "--destinations", self.destinations,
            "--destinations-id-field", "NAME",
            "--network-data-source", self.local_nd,
            "--travel-mode", self.local_tm_dist,
            "--time-units", "Minutes",
            "--distance-units", "Miles",
            "--max-routes", "15",
            "--max-processes", "4",
            "--out-routes", out_routes,
            "--reverse-direction", "false",
            "--scratch-folder", out_folder,
            "--assigned-dest-field", "StoreID",
            "--time-of-day", "20220329 16:45"
        ]
        result = subprocess.run(rt_inputs, check=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(arcpy.Exists(out_routes))

    def test_cli_many_to_many(self):
        """Test the command line interface of and launch_parallel_rt_pairs function for the many-to-many."""
        out_folder = os.path.join(self.output_folder, "CLI_Output_ManyToMany")
        os.mkdir(out_folder)
        out_routes = os.path.join(self.output_gdb, "OutCLIRoutes_ManyToMany")
        rt_inputs = [
            os.path.join(sys.exec_prefix, "python.exe"),
            os.path.join(os.path.dirname(CWD), "parallel_route_pairs.py"),
            "--pair-type", "many_to_many",
            "--origins", self.origins,
            "--origins-id-field", "ID",
            "--destinations", self.destinations,
            "--destinations-id-field", "NAME",
            "--network-data-source", self.local_nd,
            "--travel-mode", self.local_tm_dist,
            "--time-units", "Minutes",
            "--distance-units", "Miles",
            "--max-routes", "15",
            "--max-processes", "4",
            "--out-routes", out_routes,
            "--reverse-direction", "false",
            "--scratch-folder", out_folder,
            "--od-pair-table", self.od_pair_table,
            "--time-of-day", "20220329 16:45"
        ]
        result = subprocess.run(rt_inputs, check=True)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(arcpy.Exists(out_routes))


if __name__ == '__main__':
    unittest.main()
