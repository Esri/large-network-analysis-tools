"""Unit tests for the parallel_.py module.

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
import unittest
import pandas as pd
from copy import deepcopy
from glob import glob
import arcpy

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import parallel_odcm  # noqa: E402, pylint: disable=wrong-import-position
import helpers  # noqa: E402, pylint: disable=wrong-import-position
import portal_credentials  # noqa: E402, pylint: disable=wrong-import-position

TEST_ARROW = False
if helpers.arcgis_version >= "2.9":
    # The pyarrow module was not included in earlier versions of Pro, and the toArrowTable method was
    # added to the ODCostMatrix object at Pro 2.9. Do not attempt to test this output format in
    # earlier versions of Pro.
    TEST_ARROW = True
    import pyarrow as pa


class TestParallelODCM(unittest.TestCase):
    """Test cases for the parallel_odcm module."""

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
        self.portal_nd = portal_credentials.PORTAL_URL
        self.portal_tm = portal_credentials.PORTAL_TRAVEL_MODE

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.PORTAL_USERNAME, portal_credentials.PORTAL_PASSWORD)

        # Create a unique output directory and gdb for this test
        self.scratch_folder = os.path.join(
            CWD, "TestOutput", "Output_ParallelODCM_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        self.od_args = {
            "origins": self.origins,
            "destinations": self.destinations,
            "output_format": helpers.OutputFormat.featureclass,
            "output_od_location": os.path.join(self.output_gdb, "TestOutput"),
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "cutoff": 2,
            "num_destinations": 1,
            "time_of_day": None,
            "scratch_folder": self.scratch_folder,
            "barriers": []
        }

        self.parallel_od_class_args = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_dist,
            "output_format": "Feature class",
            "output_od_location": os.path.join(self.output_gdb, "TestOutput"),
            "max_origins": 1000,
            "max_destinations": 1000,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 2,
            "num_destinations": 1,
            "time_of_day": "20220329 16:45",
            "barriers": []
        }

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

    def test_ODCostMatrix_solve_featureclass(self):
        """Test the solve method of the ODCostMatrix class with feature class output."""
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

    def test_ODCostMatrix_solve_csv(self):
        """Test the solve method of the ODCostMatrix class with csv output."""
        # Initialize an ODCostMatrix analysis object
        out_folder = os.path.join(self.scratch_folder, "ODCostMatrix_CSV")
        os.mkdir(out_folder)
        od_inputs = deepcopy(self.od_args)
        od_inputs["output_format"] = helpers.OutputFormat.csv
        od_inputs["output_od_location"] = out_folder
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        # Solve a chunk
        origin_criteria = [1, 2]  # Encompasses 2 rows
        dest_criteria = [8, 12]  # Encompasses 5 rows
        od.solve(origin_criteria, dest_criteria)
        # Check results
        self.assertIsInstance(od.job_result, dict)
        self.assertTrue(od.job_result["solveSucceeded"], "OD solve failed")
        expected_out_file = os.path.join(
            out_folder,
            f"ODLines_O_{origin_criteria[0]}_{origin_criteria[1]}_D_{dest_criteria[0]}_{dest_criteria[1]}.csv"
        )
        self.assertTrue(os.path.exists(od.job_result["outputLines"]), "OD line CSV file output does not exist.")
        self.assertEqual(expected_out_file, od.job_result["outputLines"], "OD line CSV file has the wrong filepath.")
        row_count = pd.read_csv(expected_out_file).shape[0]
        self.assertEqual(2, row_count, "OD line CSV file has an incorrect number of rows.")

    @unittest.skipIf(not TEST_ARROW, "Arrow table output is not available in versions of Pro prior to 2.9.")
    def test_ODCostMatrix_solve_arrow(self):
        """Test the solve method of the ODCostMatrix class with Arrow output."""
        # Initialize an ODCostMatrix analysis object
        out_folder = os.path.join(self.scratch_folder, "ODCostMatrix_Arrow")
        os.mkdir(out_folder)
        od_inputs = deepcopy(self.od_args)
        od_inputs["output_format"] = helpers.OutputFormat.arrow
        od_inputs["output_od_location"] = out_folder
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        # Solve a chunk
        origin_criteria = [1, 2]  # Encompasses 2 rows
        dest_criteria = [8, 12]  # Encompasses 5 rows
        od.solve(origin_criteria, dest_criteria)
        # Check results
        self.assertIsInstance(od.job_result, dict)
        self.assertTrue(od.job_result["solveSucceeded"], "OD solve failed")
        expected_out_file = os.path.join(
            out_folder,
            f"ODLines_O_{origin_criteria[0]}_{origin_criteria[1]}_D_{dest_criteria[0]}_{dest_criteria[1]}.arrow"
        )
        self.assertTrue(os.path.exists(od.job_result["outputLines"]), "OD line Arrow file output does not exist.")
        self.assertEqual(expected_out_file, od.job_result["outputLines"], "OD line Arrow file has the wrong filepath.")
        with pa.memory_map(expected_out_file, 'r') as source:
            batch_reader = pa.ipc.RecordBatchFileReader(source)
            arrow_table = batch_reader.read_all()
        self.assertEqual(2, arrow_table.num_rows, "OD line Arrow file has an incorrect number of rows.")

    def test_ODCostMatrix_solve_service_featureclass(self):
        """Test the solve method of the ODCostMatrix class using a service with feature class output.

        When a service is used, the code does special extra steps to ensure the original origin and
        destination OIDs are transferred instead of reset starting with 1 for each chunk. This test
        specifically validates this special behavior.
        """
        # Initialize an ODCostMatrix analysis object
        od_inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "output_format": helpers.OutputFormat.featureclass,
            "output_od_location": os.path.join(self.output_gdb, "TestOutputService"),
            "network_data_source": self.portal_nd,
            "travel_mode": self.portal_tm,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "cutoff": 10,
            "num_destinations": None,
            "time_of_day": None,
            "scratch_folder": self.scratch_folder,
            "barriers": []
        }
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        # Solve a chunk
        origin_criteria = [45, 50]
        dest_criteria = [8, 12]
        od.solve(origin_criteria, dest_criteria)
        self.assertTrue(od.job_result["solveSucceeded"], "OD solve failed")
        self.assertTrue(arcpy.Exists(od.job_result["outputLines"]), "OD line output does not exist.")
        # Ensure the OriginOID and DestinationOID fields in the output have the correct numerical ranges
        fields_to_check = ["OriginOID", "DestinationOID"]
        for row in arcpy.da.SearchCursor(od.job_result["outputLines"], fields_to_check):
            self.assertTrue(origin_criteria[0] <= row[0] <= origin_criteria[1], f"{fields_to_check[0]} out of range.")
            self.assertTrue(dest_criteria[0] <= row[1] <= dest_criteria[1], f"{fields_to_check[1]} out of range.")

    def test_ODCostMatrix_solve_service_csv(self):
        """Test the solve method of the ODCostMatrix class using a service with csv output.

        When a service is used, the code does special extra steps to ensure the original origin and
        destination OIDs are transferred instead of reset starting with 1 for each chunk. This test
        specifically validates this special behavior.
        """
        out_folder = os.path.join(self.scratch_folder, "ODCostMatrix_CSV_service")
        os.mkdir(out_folder)
        # Initialize an ODCostMatrix analysis object
        od_inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "output_format": helpers.OutputFormat.csv,
            "output_od_location": out_folder,
            "network_data_source": self.portal_nd,
            "travel_mode": self.portal_tm,
            "time_units": arcpy.nax.TimeUnits.Minutes,
            "distance_units": arcpy.nax.DistanceUnits.Miles,
            "cutoff": 10,
            "num_destinations": None,
            "time_of_day": None,
            "scratch_folder": self.scratch_folder,
            "barriers": []
        }
        od = parallel_odcm.ODCostMatrix(**od_inputs)
        # Solve a chunk
        origin_criteria = [45, 50]
        dest_criteria = [8, 12]
        od.solve(origin_criteria, dest_criteria)
        self.assertTrue(od.job_result["solveSucceeded"], "OD solve failed")
        expected_out_file = os.path.join(
            out_folder,
            f"ODLines_O_{origin_criteria[0]}_{origin_criteria[1]}_D_{dest_criteria[0]}_{dest_criteria[1]}.csv"
        )
        self.assertTrue(os.path.exists(od.job_result["outputLines"]), "OD line CSV file output does not exist.")
        self.assertEqual(expected_out_file, od.job_result["outputLines"], "OD line CSV file has the wrong filepath.")
        # Ensure the OriginOID and DestinationOID fields in the output have the correct numerical ranges
        df = pd.read_csv(expected_out_file)
        self.assertTrue(df["OriginOID"].min() >= origin_criteria[0], "OriginOID out of range.")
        self.assertTrue(df["OriginOID"].max() <= origin_criteria[1], "OriginOID out of range.")
        self.assertTrue(df["DestinationOID"].min() >= dest_criteria[0], "DestinationOID out of range.")
        self.assertTrue(df["DestinationOID"].max() <= dest_criteria[1], "DestinationOID out of range.")

    def test_solve_od_cost_matrix(self):
        """Test the solve_od_cost_matrix function."""
        result = parallel_odcm.solve_od_cost_matrix([[1, 2], [8, 12]], self.od_args)
        # Check results
        self.assertIsInstance(result, dict)
        self.assertTrue(os.path.exists(result["logFile"]), "Log file does not exist.")
        self.assertTrue(result["solveSucceeded"], "OD solve failed")
        self.assertTrue(arcpy.Exists(result["outputLines"]), "OD line output does not exist.")
        self.assertEqual(2, int(arcpy.management.GetCount(result["outputLines"]).getOutput(0)))

    def test_ParallelODCalculator_validate_od_settings(self):
        """Test the _validate_od_settings function."""
        # Test that with good inputs, we return the correct optimized field name
        od_calculator = parallel_odcm.ParallelODCalculator(**self.parallel_od_class_args)
        optimized_cost_field = od_calculator._validate_od_settings()
        self.assertEqual("Total_Distance", optimized_cost_field)
        # Test completely invalid travel mode
        od_inputs = deepcopy(self.parallel_od_class_args)
        od_inputs["travel_mode"] = "InvalidTM"
        od_calculator = parallel_odcm.ParallelODCalculator(**od_inputs)
        error_type = ValueError if helpers.arcgis_version >= "3.1" else RuntimeError
        with self.assertRaises(error_type):
            od_calculator._validate_od_settings()

    def test_ParallelODCalculator_solve_od_in_parallel_featureclass(self):
        """Test the solve_od_in_parallel function. Output to feature class."""
        out_od_lines = os.path.join(self.output_gdb, "Out_OD_Lines")
        inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_time,
            "output_format": "Feature class",
            "output_od_location": out_od_lines,
            "max_origins": 20,
            "max_destinations": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 30,
            "num_destinations": 2,
            "time_of_day": None,
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

    def test_ParallelODCalculator_solve_od_in_parallel_featureclass_no_dest_limit(self):
        """Test the solve_od_in_parallel function. Output to feature class. No destination limit.

        A different codepath is used when post-processing OD Line feature classes if there is no destination limit.
        """
        out_od_lines = os.path.join(self.output_gdb, "Out_OD_Lines_NoLimit")
        inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_time,
            "output_format": "Feature class",
            "output_od_location": out_od_lines,
            "max_origins": 20,
            "max_destinations": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 30,
            "num_destinations": None,
            "time_of_day": None,
            "barriers": []
        }

        # Run parallel process. This calculates the OD and also post-processes the results
        od_calculator = parallel_odcm.ParallelODCalculator(**inputs)
        od_calculator.solve_od_in_parallel()

        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        self.assertEqual(4545, int(arcpy.management.GetCount(out_od_lines).getOutput(0)))

    def test_ParallelODCalculator_solve_od_in_parallel_csv(self):
        """Test the solve_od_in_parallel function. Output to CSV."""
        out_folder = os.path.join(self.scratch_folder, "ParallelODCalculator_CSV")
        os.mkdir(out_folder)
        inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_time,
            "output_format": "CSV files",
            "output_od_location": out_folder,
            "max_origins": 20,
            "max_destinations": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 30,
            "num_destinations": 2,
            "time_of_day": None,
            "barriers": []
        }

        # Run parallel process. This calculates the OD and also post-processes the results
        od_calculator = parallel_odcm.ParallelODCalculator(**inputs)
        od_calculator.solve_od_in_parallel()

        # Check results
        csv_files = glob(os.path.join(out_folder, "*.csv"))
        self.assertEqual(11, len(csv_files), "Incorrect number of CSV files produced.")
        df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)
        # With 2 destinations for each origin, expect 414 rows in the output
        # Note: 1 origin finds no destinations, and that's why we don't have 416.
        self.assertEqual(414, df.shape[0], "Incorrect number of rows in combined output CSV files.")

    @unittest.skipIf(not TEST_ARROW, "Arrow table output is not available in versions of Pro prior to 2.9.")
    def test_ParallelODCalculator_solve_od_in_parallel_arrow(self):
        """Test the solve_od_in_parallel function. Output to Arrow files."""
        out_folder = os.path.join(self.scratch_folder, "ParallelODCalculator_Arrow")
        os.mkdir(out_folder)
        inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_time,
            "output_format": "Apache Arrow files",
            "output_od_location": out_folder,
            "max_origins": 20,
            "max_destinations": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 30,
            "num_destinations": 2,
            "time_of_day": None,
            "barriers": []
        }

        # Run parallel process. This calculates the OD and also post-processes the results
        od_calculator = parallel_odcm.ParallelODCalculator(**inputs)
        od_calculator.solve_od_in_parallel()

        # Check results
        arrow_files = glob(os.path.join(out_folder, "*.arrow"))
        self.assertEqual(11, len(arrow_files), "Incorrect number of CSV files produced.")
        arrow_dfs = []
        for arrow_file in arrow_files:
            with pa.memory_map(arrow_file, 'r') as source:
                batch_reader = pa.ipc.RecordBatchFileReader(source)
                chunk_table = batch_reader.read_all()
            arrow_dfs.append(chunk_table.to_pandas(split_blocks=True))
        df = pd.concat(arrow_dfs, ignore_index=True)
        # With 2 destinations for each origin, expect 414 rows in the output
        # Note: 1 origin finds no destinations, and that's why we don't have 416.
        self.assertEqual(414, df.shape[0], "Incorrect number of rows in combined output Arrow files.")


if __name__ == '__main__':
    unittest.main()
