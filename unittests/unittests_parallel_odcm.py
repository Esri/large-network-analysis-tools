"""Unit tests for the parallel_parallel_odcm.py module.'

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
# pylint: disable=import-error, protected-access, invalid-name

import sys
import os
import datetime
import unittest
from copy import deepcopy
import arcpy

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import parallel_odcm  # noqa: E402, pylint: disable=wrong-import-position


class TestParallelODCM(unittest.TestCase):
    """Test cases for the odcm module."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        """Set up shared test properties."""
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = os.path.join(sf_gdb, "Analysis", "TractCentroids")
        self.destinations = os.path.join(sf_gdb, "Analysis", "Hospitals")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        self.local_tm_time = "Driving Time"
        self.local_tm_dist = "Driving Distance"

        # Create a unique output directory and gdb for this test
        self.output_folder = os.path.join(
            CWD, "TestOutput", "Output_ParallelODCM_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.output_folder)
        self.output_gdb = os.path.join(self.output_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        self.od_args = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "cutoff": 2,
            "num_destinations": 1,
            "output_folder": self.output_folder,
            "barriers": []
        }

        self.parallel_od_class_args = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "output_od_lines": os.path.join(self.output_gdb, "TestOutput"),
            "max_origins": 1000,
            "max_destinations": 1000,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 2,
            "num_destinations": 1,
            "barriers": []
        }

    def test_run_gp_tool(self):
        """Test the run_gp_tool function."""
        # Test for handled tool execute error (create fgdb in invalid folder)
        with self.assertRaises(arcpy.ExecuteError):
            parallel_odcm.run_gp_tool(
                arcpy.management.CreateFileGDB,
                [self.output_folder + "DoesNotExist"],
                {"out_name": "outputs.gdb"}
            )
        # Test for handled non-arcpy error when calling function
        with self.assertRaises(TypeError):
            parallel_odcm.run_gp_tool("BadTool", [self.output_folder])
        # Valid call to tool with simple function
        parallel_odcm.run_gp_tool(arcpy.management.CreateFileGDB, [self.output_folder], {"out_name": "testRunTool.gdb"})

    def test_ODCostMatrix_hour_to_time_units(self):
        """Test the _hour_to_time_units method of the ODCostMatrix class."""
        # Sanity test to make sure the method works for valid units
        od_inputs = deepcopy(self.od_args)
        od_inputs["time_units"] = arcpy.nax.TimeUnits.Seconds
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        self.assertEqual(3600, od._hour_to_time_units())

    def test_ODCostMatrix_mile_to_dist_units(self):
        """Test the _mile_to_dist_units method of the ODCostMatrix class."""
        # Sanity test to make sure the method works for valid units
        od_inputs = deepcopy(self.od_args)
        od_inputs["distance_units"] = arcpy.nax.DistanceUnits.Kilometers
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        self.assertEqual(1.60934, od._mile_to_dist_units())

    def test_ODCostMatrix_convert_time_cutoff_to_distance(self):
        """Test the _convert_time_cutoff_to_distance method of the ODCostMatrix class."""
        # We start with a 20-minute cutoff. The method converts this to a reasonable distance in units of miles.
        od_inputs = deepcopy(self.od_args)
        od_inputs["travel_mode"] = self.local_tm_time
        od_inputs["cutoff"] = 20
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        self.assertAlmostEqual(28, od._convert_time_cutoff_to_distance(), 1)

    def test_ODCostMatrix_select_inputs(self):
        """Test the _select_inputs method of the ODCostMatrix class."""
        od = parallel_odcm.ODCostMatrix(**self.od_args)
        origin_criteria = [1, 2]  # Encompasses 2 rows in the southwest corner

        # Test when a subset of destinations meets the cutoff criteria
        dest_criteria = [8, 12]  # Encompasses 5 rows. Two are close to the origins.
        od._select_inputs(origin_criteria, dest_criteria)
        self.assertEqual(2, int(arcpy.management.GetCount(od.input_origins_layer_obj).getOutput(0)))
        # Only two destinations fall within the distance threshold
        self.assertEqual(2, int(arcpy.management.GetCount(od.input_destinations_layer_obj).getOutput(0)))

        # Test when none of the destinations are within the threshold
        dest_criteria = [14, 17]  # Encompasses 4 locations in the far northeast corner
        od._select_inputs(origin_criteria, dest_criteria)
        self.assertEqual(2, int(arcpy.management.GetCount(od.input_origins_layer_obj).getOutput(0)))
        self.assertIsNone(
            od.input_destinations_layer_obj,
            "Destinations layer should be None since no destinations fall within the straight-line cutoff of origins."
        )

    def test_ODCostMatrix_solve(self):
        """Test the solve method of the ODCostMatrix class."""
        # Initialize an ODCostMatrix analysis object
        od = parallel_odcm.ODCostMatrix(**self.od_args)
        # Solve a chunk
        origin_criteria = [1, 2]  # Encompasses 2 rows
        dest_criteria = [8, 12]  # Encompasses 5 rows
        od.solve(origin_criteria, dest_criteria)
        # Check results
        self.assertIsInstance(od.job_result, dict)
        self.assertTrue(od.job_result["solveSucceeded"], "OD solve failed")
        self.assertTrue(arcpy.Exists(od.job_result["outputLines"]), "OD line output does not exist.")
        self.assertEqual(2, int(arcpy.management.GetCount(od.job_result["outputLines"]).getOutput(0)))

    def test_solve_od_cost_matrix(self):
        """Test the solve_od_cost_matrix function."""
        result = parallel_odcm.solve_od_cost_matrix(self.od_args, [[1, 2], [8, 12]])
        # Check results
        self.assertIsInstance(result, dict)
        self.assertTrue(os.path.exists(result["logFile"]), "Log file does not exist.")
        self.assertTrue(result["solveSucceeded"], "OD solve failed")
        self.assertTrue(arcpy.Exists(result["outputLines"]), "OD line output does not exist.")
        self.assertEqual(2, int(arcpy.management.GetCount(result["outputLines"]).getOutput(0)))

    def test_ParallelODCalculator_get_oid_ranges_for_input(self):
        """Test the _get_oid_ranges_for_input function."""
        od_calculator = parallel_odcm.ParallelODCalculator(**self.parallel_od_class_args)
        ranges = od_calculator._get_oid_ranges_for_input(self.parallel_od_class_args["origins"], 50)
        self.assertEqual([[1, 50], [51, 100], [101, 150], [151, 200], [201, 208]], ranges)

    def test_ParallelODCalculator_validate_od_settings(self):
        """Test the _validate_od_settings function."""
        # Test that with good inputs, we return the correct optimized field name
        od_calculator = parallel_odcm.ParallelODCalculator(**self.parallel_od_class_args)
        self.assertEqual("Total_Distance", od_calculator.optimized_cost_field)
        # Test completely invalid travel mode
        od_inputs = deepcopy(self.parallel_od_class_args)
        od_inputs["travel_mode"] = "InvalidTM"
        with self.assertRaises(RuntimeError):
            parallel_odcm.ParallelODCalculator(**od_inputs)

    def test_ParallelODCalculator_solve_od_in_parallel(self):
        """Test the solve_od_in_parallel function, which actually solves the ODs in parallel."""
        out_od_lines = os.path.join(self.output_gdb, "Out_OD_Lines")
        inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_time,
            "output_od_lines": out_od_lines,
            "max_origins": 20,
            "max_destinations": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 30,
            "num_destinations": 2,
            "barriers": []
        }
        # Run parallel process. This calculates the OD and also post-processes the results
        od_calculator = parallel_odcm.ParallelODCalculator(**inputs)
        od_calculator.solve_od_in_parallel()

        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        # With 2 destinations for each origin, expect 414 rows in the output
        # Note: 1 origin finds no destinations, and that's why we don't have 416.
        self.assertEqual(414, int(arcpy.management.GetCount(out_od_lines).getOutput(0)))


if __name__ == '__main__':
    unittest.main()
