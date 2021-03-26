############################################################################
## Toolbox name: LargeNetworkAnalysisTools
############################################################################
'''Unit tests for the SolveLargeODCostMatrix script tool. The test cases focus
on making sure the tool parameters work correctly. The tool script is tested in
more detail in unittests_odcm.py.'''
################################################################################
'''Copyright 2021 Esri
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'''
################################################################################
 
import sys
import os
import datetime
import unittest

import arcpy

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import odcm
import portal_credentials


class TestSolveLargeODCostMatrixTool(unittest.TestCase):
    '''Test cases for the SolveLargeODCostMatrix script tool.'''

    @classmethod
    def setUpClass(self):
        self.maxDiff = None

        tbx_path = os.path.join(os.path.dirname(CWD), "LargeNetworkAnalysisTools.pyt")
        arcpy.ImportToolbox(tbx_path)

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = os.path.join(sf_gdb, "Analysis", "TractCentroids")
        self.destinations = os.path.join(sf_gdb, "Analysis", "Hospitals")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        self.local_tm_time = "Driving Time"
        self.local_tm_dist = "Driving Distance"
        self.portal_nd = portal_credentials.portal_url  # Must be arcgis.com for test to work
        self.portal_tm = portal_credentials.portal_travel_mode

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.portal_username, portal_credentials.portal_password)

        # Create a unique output directory and gdb for this test
        self.output_folder = os.path.join(
            CWD, "TestOutput", "Output_Tool_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.output_folder)
        self.output_gdb = os.path.join(self.output_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

    def test_run_tool_time_units(self):
        '''Test that the tool runs with a time-based travel mode.'''
        # Run tool
        out_od_lines = os.path.join(self.output_gdb, "Time_ODLines")
        out_origins = os.path.join(self.output_gdb, "Time_Origins")
        out_destinations = os.path.join(self.output_gdb, "Time_Destinations")
        arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(
            self.origins,
            self.destinations,
            out_od_lines,
            out_origins,
            out_destinations,
            self.local_nd,
            self.local_tm_time,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            15,  # cutoff
            1,  # number of destinations
            None,  # barriers
            True  # precalculate network locations
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))

    def test_run_tool_distance_units(self):
        '''Test that the tool runs with a distance-based travel mode. Also use barriers.'''
        # Run tool
        out_od_lines = os.path.join(self.output_gdb, "Dist_ODLines")
        out_origins = os.path.join(self.output_gdb, "Dist_Origins")
        out_destinations = os.path.join(self.output_gdb, "Dist_Destinations")
        arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(
            self.origins,
            self.destinations,
            out_od_lines,
            out_origins,
            out_destinations,
            self.local_nd,
            self.local_tm_dist,
            "Minutes",
            "Miles",
            50,  # chunk size
            4,  # max processes
            5,  # cutoff
            2,  # number of destinations
            None,  # barriers
            True  # precalculate network locations
        )
        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))

    def test_agol_max_processes(self):
        '''Test for correct error when max processes exceeds the limit for AGOL.'''
        with self.assertRaises(arcpy.ExecuteError) as ex:
            # Run tool
            out_od_lines = os.path.join(self.output_gdb, "Err_ODLines")
            out_origins = os.path.join(self.output_gdb, "Err_Origins")
            out_destinations = os.path.join(self.output_gdb, "Err_Destinations")
            arcpy.LargeNetworkAnalysisTools.SolveLargeODCostMatrix(
                self.origins,
                self.destinations,
                out_od_lines,
                out_origins,
                out_destinations,
                self.portal_nd,
                self.portal_tm,
                "Minutes",
                "Miles",
                50,  # chunk size
                500,  # max processes
                15,  # cutoff
                1,  # number of destinations
                None,  # barriers
                True  # precalculate network locations
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            (
                f"The maximum number of parallel processes cannot exceed {odcm.MAX_AGOL_PROCESSES} when the "
                "ArcGIS Online services are used as the network data source."
            ),
            "Failed to execute (SolveLargeODCostMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)


if __name__ == '__main__':
    unittest.main()
