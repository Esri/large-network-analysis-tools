############################################################################
## Toolbox name: LargeNetworkAnalysisTools
############################################################################
'''Unit tests for the odcm.py module.'''
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
from copy import deepcopy

import arcpy

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CWD))
import odcm
import portal_credentials  # Contains log-in for an ArcGIS Online account to use as a test portal


class TestODCM(unittest.TestCase):
    '''Test cases for the odcm module.'''

    @classmethod
    def setUpClass(self):
        self.maxDiff = None

        self.input_data_folder = os.path.join(CWD, "TestInput")
        sf_gdb = os.path.join(self.input_data_folder, "SanFrancisco.gdb")
        self.origins = os.path.join(sf_gdb, "Analysis", "TractCentroids")
        self.destinations = os.path.join(sf_gdb, "Analysis", "Hospitals")
        self.local_nd = os.path.join(sf_gdb, "Transportation", "Streets_ND")
        self.local_tm_time = "Driving Time"
        self.local_tm_dist = "Driving Distance"
        self.portal_nd = portal_credentials.portal_url
        self.portal_tm = portal_credentials.portal_travel_mode

        arcpy.SignInToPortal(self.portal_nd, portal_credentials.portal_username, portal_credentials.portal_password)

        # Create a unique output directory and gdb for this test
        self.output_folder = os.path.join(
            CWD, "TestOutput", "Output_ODCM_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
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

    def test_run_gp_tool(self):
        '''Test the run_gp_tool function.'''
        # Test for handled tool execute error (create fgdb in invalid folder)
        with self.assertRaises(arcpy.ExecuteError):
            odcm.run_gp_tool(
                arcpy.management.CreateFileGDB,
                [self.output_folder + "DoesNotExist"],
                {"out_name": "outputs.gdb"}
            )
        # Test for handled non-arcpy error when calling function
        with self.assertRaises(TypeError):
            odcm.run_gp_tool("cabbage", [self.output_folder])
        # Valid call to tool with simple function
        odcm.run_gp_tool(arcpy.management.CreateFileGDB, [self.output_folder], {"out_name": "testRunTool.gdb"})

    def test_is_nds_service(self):
        '''Test the is_nds_service function.'''
        self.assertTrue(odcm.is_nds_service(self.portal_nd))
        self.assertFalse(odcm.is_nds_service(self.local_nd))

    def test_spatially_sort_input(self):
        '''Test the spatially_sort_input function.'''
        fc_to_sort = os.path.join(self.output_gdb, "Sorted")
        arcpy.management.Copy(self.destinations, fc_to_sort)
        odcm.spatially_sort_input(fc_to_sort, is_origins=True)
        self.assertIn("OriginOID", [f.name for f in arcpy.ListFields(fc_to_sort)])

    def test_precalculate_network_locations(self):
        '''Test the precalculate_network_locations function.'''
        loc_fields = set(["SourceID", "SourceOID", "PosAlong", "SideOfEdge"])

        # Precalculate network locations
        fc_to_precalculate = os.path.join(self.output_gdb, "Precalculated")
        arcpy.management.Copy(self.destinations, fc_to_precalculate)
        odcm.precalculate_network_locations(fc_to_precalculate, self.local_nd, self.local_tm_time)
        actual_fields = set([f.name for f in arcpy.ListFields(fc_to_precalculate)])
        self.assertTrue(loc_fields.issubset(actual_fields), "Network location fields not added")
        for row in arcpy.da.SearchCursor(fc_to_precalculate, list(loc_fields)):
            for val in row:
                self.assertIsNotNone(val)

    def test_get_tool_limits_and_is_agol(self):
        '''Test the get_tool_limits_and_is_agol function for a portal network data source.'''
        limits, is_agol = odcm.get_tool_limits_and_is_agol(self.portal_nd)
        self.assertIsInstance(limits, dict)
        self.assertIsInstance(is_agol, bool)
        self.assertIn("maximumDestinations", limits)
        self.assertIn("maximumOrigins", limits)
        if "arcgis.com" in self.portal_nd:
            # Note: If testing with some other portal, this test would need to be updated.
            self.assertTrue(is_agol)

    def test_update_max_inputs_for_service(self):
        '''Test the update_max_inputs_for_service function.'''
        tool_limits = {
            'forceHierarchyBeyondDistance': 50.0,
            'forceHierarchyBeyondDistanceUnits':
            'Miles',
            'maximumDestinations': 1000.0,
            'maximumFeaturesAffectedByLineBarriers': 500.0,
            'maximumFeaturesAffectedByPointBarriers': 250.0,
            'maximumFeaturesAffectedByPolygonBarriers': 2000.0,
            'maximumGeodesicDistanceUnitsWhenWalking': 'Miles',
            'maximumGeodesicDistanceWhenWalking': 27.0,
            'maximumOrigins': 1000.0
        }
        updated_max_origins, updated_max_destinations = odcm.update_max_inputs_for_service(1500, 700, tool_limits)
        self.assertEqual(tool_limits["maximumOrigins"], updated_max_origins)
        self.assertEqual(700, updated_max_destinations)

    def test_convert_time_units_str_to_enum(self):
        '''Test the convert_time_units_str_to_enum function.'''
        # Test all valid units
        valid_units = odcm.TIME_UNITS
        for unit in valid_units:
            enum_unit = odcm.convert_time_units_str_to_enum(unit)
            self.assertIsInstance(enum_unit, arcpy.nax.TimeUnits)
            self.assertEqual(unit.lower(), enum_unit.name.lower())
        # Test for correct error with invalid units
        bad_unit = "Lawnmower"
        with self.assertRaises(ValueError) as ex:
            odcm.convert_time_units_str_to_enum(bad_unit)
        self.assertEqual(f"Invalid time units: {bad_unit}", str(ex.exception))

    def test_convert_distance_units_str_to_enum(self):
        '''Test the convert_distance_units_str_to_enum function.'''
        # Test all valid units
        valid_units = odcm.DISTANCE_UNITS
        for unit in valid_units:
            enum_unit = odcm.convert_distance_units_str_to_enum(unit)
            self.assertIsInstance(enum_unit, arcpy.nax.DistanceUnits)
            self.assertEqual(unit.lower(), enum_unit.name.lower())
        # Test for correct error with invalid units
        bad_unit = "Weedwhacker"
        with self.assertRaises(ValueError) as ex:
            odcm.convert_distance_units_str_to_enum(bad_unit)
        self.assertEqual(f"Invalid distance units: {bad_unit}", str(ex.exception))

    def test_get_oid_ranges_for_input(self):
        '''Test the get_oid_ranges_for_input function.'''
        # Get OID ranges with no where clause
        ranges = odcm.get_oid_ranges_for_input(self.origins, 50)
        self.assertEqual([[1, 50], [51, 100], [101, 150], [151, 200], [201, 208]], ranges)

    def test_validate_od_settings(self):
        '''Test the validate_od_settings function.'''
        # Test that with good inputs, we return the correct optimized field name
        field_name = odcm.validate_od_settings(**self.od_args)
        self.assertEqual("Total_Distance", field_name)
        # Test completely invalid travel mode
        od_inputs = deepcopy(self.od_args)
        od_inputs["travel_mode"] = "Pizza"
        with self.assertRaises(RuntimeError):
            odcm.validate_od_settings(**od_inputs)

    def test_ODCostMatrix_hour_to_time_units(self):
        '''Test the _hour_to_time_units method of the ODCostMatrix class.'''
        # Sanity test to make sure the method works for valid units
        od_inputs = deepcopy(self.od_args)
        od_inputs["time_units"] = arcpy.nax.TimeUnits.Seconds
        od = odcm.ODCostMatrix(**od_inputs)
        self.assertEqual(3600, od._hour_to_time_units())

    def test_ODCostMatrix_mile_to_dist_units(self):
        '''Test the _mile_to_dist_units method of the ODCostMatrix class.'''
        # Sanity test to make sure the method works for valid units
        od_inputs = deepcopy(self.od_args)
        od_inputs["distance_units"] = arcpy.nax.DistanceUnits.Kilometers
        od = odcm.ODCostMatrix(**od_inputs)
        self.assertEqual(1.60934, od._mile_to_dist_units())

    def test_ODCostMatrix_convert_time_cutoff_to_distance(self):
        '''Test the _convert_time_cutoff_to_distance method of the ODCostMatrix class.'''
        # We start with a 20-minute cutoff. The method converts this to a reasonable distance in units of miles.
        od_inputs = deepcopy(self.od_args)
        od_inputs["travel_mode"] = self.local_tm_time
        od_inputs["cutoff"] = 20
        od = odcm.ODCostMatrix(**od_inputs)
        self.assertAlmostEqual(28, od._convert_time_cutoff_to_distance(), 1)

    def test_ODCostMatrix_select_inputs(self):
        '''Test the _select_inputs method of the ODCostMatrix class.'''
        od = odcm.ODCostMatrix(**self.od_args)
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
        '''Test the solve method of the ODCostMatrix class.'''
        # Initialize an ODCostMatrix analysis object
        od = odcm.ODCostMatrix(**self.od_args)
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
        '''Test the solve_od_cost_matrix function.'''
        result = odcm.solve_od_cost_matrix(self.od_args, [[1, 2], [8, 12]])
        # Check results
        self.assertIsInstance(result, dict)
        self.assertTrue(os.path.exists(result["logFile"]), "Log file does not exist.")
        self.assertTrue(result["solveSucceeded"], "OD solve failed")
        self.assertTrue(arcpy.Exists(result["outputLines"]), "OD line output does not exist.")
        self.assertEqual(2, int(arcpy.management.GetCount(result["outputLines"]).getOutput(0)))

    def test_compute_ods_in_parallel(self):
        '''Test the compute_ods_in_parallel function, which actually solves the ODs in parallel.'''
        # Run parallel process
        out_od_lines = os.path.join(self.output_gdb, "Out_OD_Lines")
        out_origins = os.path.join(self.output_gdb, "Out_Origins")
        out_destinations = os.path.join(self.output_gdb, "Out_Destinations")
        inputs = {
            "origins": self.origins,
            "destinations": self.destinations,
            "output_od_lines": out_od_lines,
            "output_origins": out_origins,
            "output_destinations": out_destinations,
            "network_data_source": self.local_nd,
            "travel_mode": self.local_tm_time,
            "chunk_size": 20,
            "max_processes": 4,
            "time_units": "Minutes",
            "distance_units": "Miles",
            "cutoff": 30,
            "num_destinations": 2,
            "precalculate_network_locations": True,
        }
        odcm.compute_ods_in_parallel(**inputs)

        # Check results
        self.assertTrue(arcpy.Exists(out_od_lines))
        self.assertTrue(arcpy.Exists(out_origins))
        self.assertTrue(arcpy.Exists(out_destinations))
        # With 2 destinations for each origin, expect 414 rows in the output
        # Note: 1 origin finds no destinations, and that's why we don't have 416.
        self.assertEqual(414, int(arcpy.management.GetCount(out_od_lines).getOutput(0)))


if __name__ == '__main__':
    unittest.main()
