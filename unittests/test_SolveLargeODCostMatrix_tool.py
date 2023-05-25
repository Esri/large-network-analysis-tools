"""Unit tests for the SolveLargeODCostMatrix script tool. The test cases focus
on making sure the tool parameters work correctly.

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
# pylint: disable=import-error, invalid-name

import sys
import os
import datetime
from glob import glob
import unittest
import arcpy
import portal_credentials

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import helpers  # noqa: E402, pylint: disable=wrong-import-position


class TestSolveLargeODCostMatrixTool(unittest.TestCase):
    """Test cases for the SolveLargeODCostMatrix script tool."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        self.maxDiff = None

        tbx_path = os.path.join(os.path.dirname(CWD), "LargeNetworkAnalysisTools.pyt")
        arcpy.ImportToolbox(tbx_path)

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = os.path.join(sf_gdb, "Analysis", "TractCentroids")
        self.destinations = os.path.join(sf_gdb, "Analysis", "Hospitals")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        tms = arcpy.nax.GetTravelModes(self.local_nd)
        self.local_tm_time = tms["Driving Time"]
        self.local_tm_dist = tms["Driving Distance"]
        self.portal_nd = portal_credentials.PORTAL_URL  # Must be arcgis.com for test to work
        self.portal_tm = portal_credentials.PORTAL_TRAVEL_MODE

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.PORTAL_USERNAME, portal_credentials.PORTAL_PASSWORD)

        # Create a unique output directory and gdb for this test
        self.scratch_folder = os.path.join(
            CWD, "TestOutput",
            "Output_SolveLargeODCostMatrix_Tool_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        # Copy some data to the output gdb to serve as barriers. Do not use tutorial data directly as input because
        # the tests will write network location fields to it, and we don't want to modify the user's original data.
        self.barriers = os.path.join(self.output_gdb, "Barriers")
        arcpy.management.Copy(os.path.join(sf_gdb, "Analysis", "CentralDepots"), self.barriers)

    def test_run_tool_time_units_feature_class(self):
        """Test that the tool runs with a time-based travel mode. Write output to a feature class."""
        # Run tool
        out_od_lines = os.path.join(self.output_gdb, "Time_ODLines")
        out_origins = os.path.join(self.output_gdb, "Time_Origins")
        out_destinations = os.path.join(self.output_gdb, "Time_Destinations")
        arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
            self.origins,
            self.destinations,
            self.local_nd,
            self.local_tm_time,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            out_origins,
            out_destinations,
            "Feature class",
            out_od_lines,
            None,
            15,  # cutoff
            1,  # number of destinations
            datetime.datetime(2022, 3, 29, 16, 45, 0),  # time of day
            None,  # barriers
            True,  # precalculate network locations
            True  # Spatially sort inputs
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))

    def test_run_tool_distance_units_csv_barriers(self):
        """Test that the tool runs with a distance-based travel mode. Write output to CSVs. Also use barriers."""
        # Run tool
        out_origins = os.path.join(self.output_gdb, "Dist_Origins")
        out_destinations = os.path.join(self.output_gdb, "Dist_Destinations")
        out_folder = os.path.join(self.scratch_folder, "DistUnits_CSVs")
        arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
            self.origins,
            self.destinations,
            self.local_nd,
            self.local_tm_dist,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            out_origins,
            out_destinations,
            "CSV files",
            None,
            out_folder,
            5,  # cutoff
            2,  # number of destinations
            None,  # time of day
            self.barriers,  # barriers
            True,  # precalculate network locations
            True  # Spatially sort inputs
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))
        csv_files = glob(os.path.join(out_folder, "*.csv"))
        self.assertGreater(len(csv_files), 0)

    @unittest.skipIf(
        helpers.arcgis_version < "2.9", "Arrow table output is not available in versions of Pro prior to 2.9.")
    def test_run_tool_time_units_arrow(self):
        """Test the tool with Apache Arrow outputs."""
        # Run tool
        out_origins = os.path.join(self.output_gdb, "Arrow_Origins")
        out_destinations = os.path.join(self.output_gdb, "Arrow_Destinations")
        out_folder = os.path.join(self.scratch_folder, "ArrowOutputs")
        arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
            self.origins,
            self.destinations,
            self.local_nd,
            self.local_tm_dist,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            out_origins,
            out_destinations,
            "Apache Arrow files",
            None,
            out_folder,
            5,  # cutoff
            2,  # number of destinations
            None,  # time of day
            None,  # barriers
            True,  # precalculate network locations
            True  # Spatially sort inputs
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))
        csv_files = glob(os.path.join(out_folder, "*.arrow"))
        self.assertGreater(len(csv_files), 0)

    def test_run_tool_service(self):
        """Test that the tool runs with a service as a network data source."""
        out_od_lines = os.path.join(self.output_gdb, "Service_ODLines")
        out_origins = os.path.join(self.output_gdb, "Service_Origins")
        out_destinations = os.path.join(self.output_gdb, "Service_Destinations")
        arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
            self.origins,
            self.destinations,
            self.portal_nd,
            self.portal_tm,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            out_origins,
            out_destinations,
            "Feature class",
            out_od_lines,
            None,
            15,  # cutoff
            1,  # number of destinations
            None,  # time of day
            None,  # barriers
            False,  # precalculate network locations
            True  # Spatially sort inputs
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))

    def test_error_required_output_od_lines(self):
        """Test for correct error when output format is Feature class and output OD Lines not specified."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            # Run tool
            out_origins = os.path.join(self.output_gdb, "Err_Origins")
            out_destinations = os.path.join(self.output_gdb, "Err_Destinations")
            arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
                self.origins,
                self.destinations,
                self.local_nd,
                self.local_tm_time,
                "Minutes",
                "Miles",
                50,  # chunk size
                4,  # max processes
                out_origins,
                out_destinations,
                "Feature class",
                None,
                "Junk",
                15,  # cutoff
                1,  # number of destinations
                None,  # time of day
                None,  # barriers
                True,  # precalculate network locations
                True  # Spatially sort inputs
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            "ERROR 000735: Output OD Lines Feature Class: Value is required",
            "Failed to execute (SolveLargeODCostMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)

    def test_error_required_output_folder(self):
        """Test for correct error when output format is Feature class and output OD Lines not specified."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            # Run tool
            out_origins = os.path.join(self.output_gdb, "Err_Origins")
            out_destinations = os.path.join(self.output_gdb, "Err_Destinations")
            arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
                self.origins,
                self.destinations,
                self.local_nd,
                self.local_tm_time,
                "Minutes",
                "Miles",
                50,  # chunk size
                4,  # max processes
                out_origins,
                out_destinations,
                "CSV files",
                "Junk",
                None,
                15,  # cutoff
                1,  # number of destinations
                None,  # time of day
                None,  # barriers
                True,  # precalculate network locations
                True  # Spatially sort inputs
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            "ERROR 000735: Output Folder: Value is required",
            "Failed to execute (SolveLargeODCostMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)

    @unittest.skipIf(
        helpers.arcgis_version < "2.9", "Arrow table output is not available in versions of Pro prior to 2.9.")
    def test_error_arrow_service(self):
        """Test for correct error when the network data source is a service and requesting Arrow output."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            # Run tool
            out_od_folder = os.path.join(self.scratch_folder, "Err")
            out_origins = os.path.join(self.output_gdb, "Err_Origins")
            out_destinations = os.path.join(self.output_gdb, "Err_Destinations")
            arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(  # pylint: disable=no-member
                self.origins,
                self.destinations,
                self.portal_nd,
                self.portal_tm,
                "Minutes",
                "Miles",
                50,  # chunk size
                4,  # max processes
                out_origins,
                out_destinations,
                "Apache Arrow files",
                None,
                out_od_folder,
                15,  # cutoff
                1,  # number of destinations
                None,  # time of day
                None,  # barriers
                True,  # precalculate network locations
                True  # Spatially sort inputs
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            "Apache Arrow files output format is not available when a service is used as the network data source.",
            "Failed to execute (SolveLargeODCostMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)


if __name__ == '__main__':
    unittest.main()
