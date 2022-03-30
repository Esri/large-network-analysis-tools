"""Unit tests for the helpers.py module.'

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
import sys
import os
import datetime
import unittest
import arcpy
import portal_credentials  # Contains log-in for an ArcGIS Online account to use as a test portal

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import helpers  # noqa: E402, pylint: disable=wrong-import-position
from od_config import OD_PROPS  # noqa: E402, pylint: disable=wrong-import-position
from rt_config import RT_PROPS  # noqa: E402, pylint: disable=wrong-import-position

class TestHelpers(unittest.TestCase):
    """Test cases for the helpers module."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        """Set up shared test properties."""
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        self.sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.local_nd = os.path.join(self.sf_gdb, "Transportation", "Streets_ND")
        self.portal_nd = portal_credentials.PORTAL_URL

        self.scratch_folder = os.path.join(
            CWD, "TestOutput", "Output_Helpers_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(self.scratch_folder)
        self.output_gdb = os.path.join(self.scratch_folder, "outputs.gdb")
        arcpy.management.CreateFileGDB(os.path.dirname(self.output_gdb), os.path.basename(self.output_gdb))

    def test_is_nds_service(self):
        """Test the is_nds_service function."""
        self.assertTrue(helpers.is_nds_service(self.portal_nd))
        self.assertFalse(helpers.is_nds_service(self.local_nd))

    def test_convert_time_units_str_to_enum(self):
        """Test the convert_time_units_str_to_enum function."""
        # Test all valid units
        valid_units = helpers.TIME_UNITS
        for unit in valid_units:
            enum_unit = helpers.convert_time_units_str_to_enum(unit)
            self.assertIsInstance(enum_unit, arcpy.nax.TimeUnits)
            self.assertEqual(unit.lower(), enum_unit.name.lower())
        # Test for correct error with invalid units
        bad_unit = "BadUnit"
        with self.assertRaises(ValueError) as ex:
            helpers.convert_time_units_str_to_enum(bad_unit)
        self.assertEqual(f"Invalid time units: {bad_unit}", str(ex.exception))

    def test_convert_distance_units_str_to_enum(self):
        """Test the convert_distance_units_str_to_enum function."""
        # Test all valid units
        valid_units = helpers.DISTANCE_UNITS
        for unit in valid_units:
            enum_unit = helpers.convert_distance_units_str_to_enum(unit)
            self.assertIsInstance(enum_unit, arcpy.nax.DistanceUnits)
            self.assertEqual(unit.lower(), enum_unit.name.lower())
        # Test for correct error with invalid units
        bad_unit = "BadUnit"
        with self.assertRaises(ValueError) as ex:
            helpers.convert_distance_units_str_to_enum(bad_unit)
        self.assertEqual(f"Invalid distance units: {bad_unit}", str(ex.exception))

    def test_convert_output_format_str_to_enum(self):
        """Test the convert_output_format_str_to_enum function."""
        # Test all valid formats
        valid_formats = helpers.OUTPUT_FORMATS
        for fm in valid_formats:
            enum_format = helpers.convert_output_format_str_to_enum(fm)
            self.assertIsInstance(enum_format, helpers.OutputFormat)
        # Test for correct error with an invalid format type
        bad_format = "BadFormat"
        with self.assertRaises(ValueError) as ex:
            helpers.convert_output_format_str_to_enum(bad_format)
        self.assertEqual(f"Invalid output format: {bad_format}", str(ex.exception))

    def test_validate_input_feature_class(self):
        """Test the validate_input_feature_class function."""
        # Test when the input feature class does note exist.
        input_fc = os.path.join(self.sf_gdb, "DoesNotExist")
        with self.subTest(feature_class=input_fc):
            with self.assertRaises(ValueError) as ex:
                helpers.validate_input_feature_class(input_fc)
            self.assertEqual(f"Input dataset {input_fc} does not exist.", str(ex.exception))

        # Test when the input feature class is empty
        input_fc = os.path.join(self.output_gdb, "EmptyFC")
        with self.subTest(feature_class=input_fc):
            arcpy.management.CreateFeatureclass(self.output_gdb, os.path.basename(input_fc))
            with self.assertRaises(ValueError) as ex:
                helpers.validate_input_feature_class(input_fc)
            self.assertEqual(f"Input dataset {input_fc} has no rows.", str(ex.exception))

    def test_precalculate_network_locations(self):
        """Test the precalculate_network_locations function."""
        loc_fields = {"SourceID", "SourceOID", "PosAlong", "SideOfEdge"}
        inputs = os.path.join(self.sf_gdb, "Analysis", "CentralDepots")

        def check_precalculated_locations(fc):
            """Check precalculated locations."""
            actual_fields = set([f.name for f in arcpy.ListFields(fc)])
            self.assertTrue(loc_fields.issubset(actual_fields), "Network location fields not added")
            for row in arcpy.da.SearchCursor(fc, list(loc_fields)):  # pylint: disable=no-member
                for val in row:
                    self.assertIsNotNone(val)

        # Precalculate locations for OD
        fc_to_precalculate = os.path.join(self.output_gdb, "Precalculated_OD")
        arcpy.management.Copy(inputs, fc_to_precalculate)
        helpers.precalculate_network_locations(fc_to_precalculate, self.local_nd, "Driving Time", OD_PROPS)
        check_precalculated_locations(fc_to_precalculate)

        # Precalculate locations for Route
        fc_to_precalculate = os.path.join(self.output_gdb, "Precalculated_Route")
        arcpy.management.Copy(inputs, fc_to_precalculate)
        helpers.precalculate_network_locations(fc_to_precalculate, self.local_nd, "Driving Time", RT_PROPS)
        check_precalculated_locations(fc_to_precalculate)

    def test_parse_std_and_write_to_gp_ui(self):
        """Test the parse_std_and_write_to_gp_ui function."""
        # There is nothing much to test here except that nothing terrible happens.
        msgs = [
            f"CRITICAL{helpers.MSG_STR_SPLITTER}Critical message",
            f"ERROR{helpers.MSG_STR_SPLITTER}Error message",
            f"WARNING{helpers.MSG_STR_SPLITTER}Warning message",
            f"INFO{helpers.MSG_STR_SPLITTER}Info message",
            f"DEBUG{helpers.MSG_STR_SPLITTER}Debug message",
            "Poorly-formatted message 1",
            f"Poorly-formatted{helpers.MSG_STR_SPLITTER}message 2"
        ]
        for msg in msgs:
            with self.subTest(msg=msg):
                helpers.parse_std_and_write_to_gp_ui(msg)


if __name__ == '__main__':
    unittest.main()
