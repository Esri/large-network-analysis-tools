"""Unit tests for the ParallelCalculateLocations script tool. The test cases focus
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

import os
import datetime
import unittest
import arcpy

CWD = os.path.dirname(os.path.abspath(__file__))


class TestSolveLargeODCostMatrixTool(unittest.TestCase):
    """Test cases for the SolveLargeODCostMatrix script tool."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        self.maxDiff = None

        tbx_path = os.path.join(os.path.dirname(CWD), "LargeNetworkAnalysisTools.pyt")
        arcpy.ImportToolbox(tbx_path)

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.test_points = os.path.join(sf_gdb, "Analysis", "TractCentroids")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        tms = arcpy.nax.GetTravelModes(self.local_nd)
        self.local_tm_time = tms["Driving Time"]

        # Create a unique output directory and gdb for this test
        self.scratch_folder = os.path.join(
            CWD, "TestOutput",
            "Output_ParallelCalcLocs_Tool_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

    def test_tool_defaults(self):
        """Test that the tool runs and works with only required parameters."""
        # The input feature class should not be overwritten by this tool, but copy it first just in case.
        fc_to_precalculate = os.path.join(self.output_gdb, "PrecalcFC_InputD")
        arcpy.management.Copy(self.test_points, fc_to_precalculate)
        output_fc = os.path.join(self.output_gdb, "PrecalcFC_OutputD")
        arcpy.LargeNetworkAnalysisTools.ParallelCalculateLocations(  # pylint: disable=no-member
            fc_to_precalculate,
            output_fc,
            self.local_nd,
        )
        self.assertTrue(arcpy.Exists(output_fc), "Output feature class does not exist")

    def test_tool_nondefaults(self):
        """Test that the tool runs and works with all input parameters."""
        # The input feature class should not be overwritten by this tool, but copy it first just in case.
        fc_to_precalculate = os.path.join(self.output_gdb, "PrecalcFC_InputND")
        arcpy.management.Copy(self.test_points, fc_to_precalculate)
        output_fc = os.path.join(self.output_gdb, "PrecalcFC_OutputND")
        arcpy.LargeNetworkAnalysisTools.ParallelCalculateLocations(  # pylint: disable=no-member
            fc_to_precalculate,
            output_fc,
            self.local_nd,
            30,
            4,
            self.local_tm_time,
            "5000 Meters",
            ["Streets"],
            [["Streets", "ObjectID <> 1"]]
        )
        self.assertTrue(arcpy.Exists(output_fc), "Output feature class does not exist")

    def test_error_invalid_query_sources(self):
        """Test for correct error when an invalid network source name is specified in the search query."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.LargeNetworkAnalysisTools.ParallelCalculateLocations(  # pylint: disable=no-member
                self.test_points,
                os.path.join(self.output_gdb, "Junk"),
                self.local_nd,
                30,
                4,
                self.local_tm_time,
                "5000 Meters",
                ["Streets"],
                [["Streets", "ObjectID <> 1"], ["BadSourceName", "ObjectID <> 2"]]  # Invalid source name
            )
        actual_messages = str(ex.exception)
        # Check for expected GP message
        self.assertIn("30254", actual_messages)

    def test_error_duplicate_query_sources(self):
        """Test for correct error when a network source is specified more than once in the search query."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.LargeNetworkAnalysisTools.ParallelCalculateLocations(  # pylint: disable=no-member
                self.test_points,
                os.path.join(self.output_gdb, "Junk"),
                self.local_nd,
                30,
                4,
                self.local_tm_time,
                "5000 Meters",
                ["Streets"],
                [["Streets", "ObjectID <> 1"], ["Streets", "ObjectID <> 2"]]  # Duplicate query
            )
        actual_messages = str(ex.exception)
        # Check for expected GP message
        self.assertIn("30255", actual_messages)

    def test_error_invalid_query(self):
        """Test for correct error when an invalid search query is specified."""
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.LargeNetworkAnalysisTools.ParallelCalculateLocations(  # pylint: disable=no-member
                self.test_points,
                os.path.join(self.output_gdb, "Junk"),
                self.local_nd,
                30,
                4,
                self.local_tm_time,
                "5000 Meters",
                ["Streets"],
                [["Streets", "NAME = 1"]]  # Bad query syntax
            )
        actual_messages = str(ex.exception)
        # Check for expected validation message
        self.assertIn("An invalid SQL statement was used", actual_messages)


if __name__ == '__main__':
    unittest.main()
