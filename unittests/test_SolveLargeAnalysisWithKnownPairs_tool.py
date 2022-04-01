"""Unit tests for the SolveLargeAnalysisWithKnownPairs script tool. The test cases focus
on making sure the tool parameters work correctly.

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


class TestSolveLargeAnalysisWithKnownPairsTool(unittest.TestCase):
    """Test cases for the SolveLargeAnalysisWithKnownPairs script tool."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        self.maxDiff = None

        tbx_path = os.path.join(os.path.dirname(CWD), "LargeNetworkAnalysisTools.pyt")
        arcpy.ImportToolbox(tbx_path)

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = os.path.join(sf_gdb, "Analysis", "TractCentroids_wStoreID")
        self.destinations = os.path.join(sf_gdb, "Analysis", "Stores")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        tms = arcpy.nax.GetTravelModes(self.local_nd)
        self.local_tm_time = tms["Driving Time"]
        self.portal_nd = portal_credentials.PORTAL_URL  # Must be arcgis.com for test to work
        self.portal_tm = portal_credentials.PORTAL_TRAVEL_MODE

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.PORTAL_USERNAME, portal_credentials.PORTAL_PASSWORD)

        # Create a unique output directory and gdb for this test
        self.scratch_folder = os.path.join(
            CWD, "TestOutput",
            "Output_SolveLargeAnalysisWithKnownPairs_Tool_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

        # Copy some data to the output gdb to serve as barriers. Do not use tutorial data directly as input because
        # the tests will write network location fields to it, and we don't want to modify the user's original data.
        self.barriers = os.path.join(self.output_gdb, "Barriers")
        arcpy.management.Copy(os.path.join(sf_gdb, "Analysis", "CentralDepots"), self.barriers)

    def test_run_tool(self):
        """Test that the tool runs with all inputs."""
        # Run tool
        out_routes = os.path.join(self.output_gdb, "OutRoutesLocal")
        arcpy.LargeNetworkAnalysisTools.SolveLargeAnalysisWithKnownPairs(  # pylint: disable=no-member
            self.origins,
            "ID",
            "StoreID",
            self.destinations,
            "NAME",
            self.local_nd,
            self.local_tm_time,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            out_routes,
            datetime.datetime(2022, 3, 29, 16, 45, 0),  # time of day
            self.barriers,  # barriers
            True,  # precalculate network locations
            True  # Sort origins
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_routes))

    def test_run_tool_service(self):
        """Test that the tool runs with a service as a network data source."""
        out_routes = os.path.join(self.output_gdb, "OutRoutesService")
        arcpy.LargeNetworkAnalysisTools.SolveLargeAnalysisWithKnownPairs(  # pylint: disable=no-member
            self.origins,
            "ID",
            "StoreID",
            self.destinations,
            "NAME",
            self.portal_nd,
            self.portal_tm,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            out_routes,
            None,  # time of day
            None,  # barriers
            False,  # precalculate network locations
            False  # Sort origins
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_routes))

    def test_error_agol_max_processes(self):
        """Test for correct error when max processes exceeds the limit for AGOL."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            # Run tool
            out_routes = os.path.join(self.output_gdb, "Junk")
            arcpy.LargeNetworkAnalysisTools.SolveLargeAnalysisWithKnownPairs(  # pylint: disable=no-member
                self.origins,
                "ID",
                "StoreID",
                self.destinations,
                "NAME",
                self.portal_nd,
                self.portal_tm,
                "Minutes",
                "Miles",
                50,  # chunk size
                2000,  # max processes
                out_routes,
                None,  # time of day
                None,  # barriers
                True,  # precalculate network locations
                True  # Sort origins
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            (
                f"The maximum number of parallel processes cannot exceed {helpers.MAX_AGOL_PROCESSES} when the "
                "ArcGIS Online services are used as the network data source."
            ),
            "Failed to execute (SolveLargeAnalysisWithKnownPairs)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)


if __name__ == '__main__':
    unittest.main()
