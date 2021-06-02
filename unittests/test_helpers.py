"""Unit tests for the helpers.py module.'

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
import sys
import os
import unittest
import arcpy
import portal_credentials  # Contains log-in for an ArcGIS Online account to use as a test portal

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import helpers  # noqa: E402, pylint: disable=wrong-import-position


class TestHelpers(unittest.TestCase):
    """Test cases for the helpers module."""

    @classmethod
    def setUpClass(self):  # pylint: disable=bad-classmethod-argument
        """Set up shared test properties."""
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        self.portal_nd = portal_credentials.PORTAL_URL

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
