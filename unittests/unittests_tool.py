############################################################################
## Toolbox name: NetworkAdequacyTools
## Created by: Melinda Morang and Jhonatan Garrido-Lecca, Esri
############################################################################
'''Unit tests for the GenerateTravelTimeMatrix script tool.'''
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


class TestGenerateTravelTimeMatrixTool(unittest.TestCase):
    '''Test cases for the GenerateTravelTimeMatrix script tool.'''

    @classmethod
    def setUpClass(self):
        self.maxDiff = None

        tbx_path = os.path.join(os.path.dirname(CWD), "NetworkAdequacyTools.pyt")
        arcpy.ImportToolbox(tbx_path)

        self.input_data_folder = os.path.join(CWD, "TestInput")
        self.input_gdb = os.path.join(self.input_data_folder, "inputs.gdb")
        self.members = os.path.join(self.input_gdb, "Members")
        self.providers = os.path.join(self.input_gdb, "Providers")
        self.local_nd = os.path.join(self.input_data_folder, "SanDiego_TravelModes.gdb", "Transportation", "Streets_ND")
        self.local_tm = "Driving Time"
        self.portal_nd = portal_credentials.portal_url  # Must be arcgis.com for test to work
        self.portal_tm = portal_credentials.portal_travel_mode
        self.specialty_field = "Specialty"
        self.geography_field = "County"

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.portal_username, portal_credentials.portal_password)

        self.output_folder = os.path.join(CWD, "TestOutput", "Output_Tool_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.output_folder)
        self.output_gdb = os.path.join(self.output_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

    def check_tool_output(self, tool_output_str, specialties_to_analyze, out_gdb_name):
        """Check that the tool has produced the expected output tables and returns the list of them in getOutput()."""
        self.assertIsInstance(tool_output_str, str)
        # Parse the tool output string to get a list of tables added to the tool's output
        tool_output_tables = sorted(tool_output_str.split(";"))
        expected_output_tables = []
        # We know what outputs should have been produced. Make sure they exist.
        for specialty in specialties_to_analyze:
            out_table = os.path.join(self.output_folder, out_gdb_name + ".gdb", f"ODStats_{specialty}")
            self.assertTrue(arcpy.Exists(out_table))
            expected_output_tables.append(out_table)
        # Check that the derived output reported by getOutput(0) matches the list of expected tables.
        expected_output_tables = sorted(expected_output_tables)
        self.assertEqual(expected_output_tables, tool_output_tables)

    def test_run_tool_time_units(self):
        '''Test that the tool runs with a time-based travel mode.'''
        # Copy members so we don't overwrite inputs
        members = os.path.join(self.output_gdb, "Members_run_tool_time_units")
        arcpy.management.Copy(self.members, members)
        # Run tool
        out_gdb_name = "Output_TimeBased"
        specialties_to_analyze = ["Primary", "Orthopedic"]
        tool_result = arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
            members,
            self.geography_field,
            self.providers,
            self.specialty_field,
            specialties_to_analyze,
            self.output_folder,
            out_gdb_name,
            self.local_nd,
            self.local_tm,
            "Minutes",
            "Miles",
            4,
            4
        )
        # Check output summarized OD tables for each specialty
        self.check_tool_output(tool_result.getOutput(0), specialties_to_analyze, out_gdb_name)
        # Check that the updated Members layer is returned by the tool
        self.assertEqual(members, tool_result.getOutput(1))
        # Check that the output summary table is produced and returned by the tool
        out_summary_table = os.path.join(self.output_folder, out_gdb_name + ".gdb", "SummarizedCompliance")
        self.assertTrue(arcpy.Exists(out_summary_table))
        self.assertEqual(out_summary_table, tool_result.getOutput(2))

    def test_run_tool_distance_units(self):
        '''Test that the tool runs with a distance-based travel mode. Also use barriers.'''
        # Copy members so we don't overwrite inputs
        members = os.path.join(self.output_gdb, "Members_run_tool_distance_units")
        arcpy.management.Copy(self.members, members)
        # Run tool
        out_gdb_name = "Output_DistanceBased"
        specialties_to_analyze = ["Dental", "Orthopedic"]
        tool_result = arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
            members,
            self.geography_field,
            self.providers,
            self.specialty_field,
            specialties_to_analyze,
            self.output_folder,
            out_gdb_name,
            self.local_nd,
            "Driving Distance",
            "Minutes",
            "Miles",
            4,
            4,
            [
                os.path.join(self.input_gdb, "PointBarriers"),
                os.path.join(self.input_gdb, "LineBarriers"),
                os.path.join(self.input_gdb, "PolygonBarriers")
            ],
            False
        )
        # Check output summarized OD tables for each specialty
        self.check_tool_output(tool_result.getOutput(0), specialties_to_analyze, out_gdb_name)
        # Check that the updated Members layer is returned by the tool
        self.assertEqual(members, tool_result.getOutput(1))
        # Check that the output summary table is produced and returned by the tool
        out_summary_table = os.path.join(self.output_folder, out_gdb_name + ".gdb", "SummarizedCompliance")
        self.assertTrue(arcpy.Exists(out_summary_table))
        self.assertEqual(out_summary_table, tool_result.getOutput(2))

    def test_bad_specialty_field(self):
        '''Test for correct error when specialty field does not exist.'''
        bad_field = "DoesNotExist"
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
                self.members,
                self.geography_field,
                self.providers,
                bad_field,
                ["Primary"],
                self.output_folder,
                "BadSpecialtyField",
                self.local_nd,
                self.local_tm,
                "Minutes",
                "Miles",
                4,
                4
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            f"ERROR 000728: Field {bad_field} does not exist within table",
            "Failed to execute (GenerateTravelTimeMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)

    def test_bad_geography_field(self):
        '''Test for correct error when geography field does not exist.'''
        bad_field = "Cooouuunntteeee"
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
                self.members,
                bad_field,
                self.providers,
                self.specialty_field,
                ["Primary"],
                self.output_folder,
                "BadSpecialtyField",
                self.local_nd,
                self.local_tm,
                "Minutes",
                "Miles",
                4,
                4
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            f"ERROR 000728: Field {bad_field} does not exist within table",
            "Failed to execute (GenerateTravelTimeMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)

    def test_gdb_already_exists(self):
        '''Test for correct error when output gdb already exists'''
        # Make a gdb so it's already there.
        gdb_name = "GDB_Exists"
        arcpy.management.CreateFileGDB(self.output_folder, gdb_name)
        # Call the tool and designate the same gdb output
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
                self.members,
                self.geography_field,
                self.providers,
                self.specialty_field,
                ["Primary"],
                self.output_folder,
                gdb_name,
                self.local_nd,
                self.local_tm,
                "Minutes",
                "Miles",
                4,
                4
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            "Output geodatabase already exists.",
            "Failed to execute (GenerateTravelTimeMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)

    def test_agol_max_processes(self):
        '''Test for correct error when max processes exceeds the limit for AGOL.'''
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
                self.members,
                self.geography_field,
                self.providers,
                self.specialty_field,
                ["Primary"],
                self.output_folder,
                "Junk",
                self.portal_nd,
                self.portal_tm,
                "Minutes",
                "Miles",
                4,
                6
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            (
                f"The maximum number of parallel processes cannot exceed {odcm.MAX_AGOL_PROCESSES} when the "
                "ArcGIS Online services are used as the network data source."
            ),
            "Failed to execute (GenerateTravelTimeMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)

    def test_other_tm_units(self):
        '''Test for correct error when the travel mode's impedance units are Other.'''
        with self.assertRaises(arcpy.ExecuteError) as ex:
            arcpy.NetworkAdequacyTools.GenerateTravelTimeMatrix(
                self.members,
                self.geography_field,
                self.providers,
                self.specialty_field,
                ["Primary"],
                self.output_folder,
                "Junk",
                self.local_nd,
                "Driving Other",
                "Minutes",
                "Miles",
                4,
                4
            )
        expected_messages = [
            "Failed to execute. Parameters are not valid.",
            "The impedance units of the selected travel mode are neither time nor distance based.",
            "Failed to execute (GenerateTravelTimeMatrix)."
        ]
        actual_messages = str(ex.exception).strip().split("\n")
        self.assertEqual(expected_messages, actual_messages)


if __name__ == '__main__':
    unittest.main()
