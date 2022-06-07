"""Helper for unit tests to create required inputs.

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
import os
import arcpy


def get_tract_centroids_with_store_id_fc(sf_gdb):
    """Create the TractCentroids_wStoreID feature class in the SanFrancisco.gdb/Analysis for use in unit tests."""
    new_fc = os.path.join(sf_gdb, "Analysis", "TractCentroids_wStoreID")
    if arcpy.Exists(new_fc):
        # The feature class exists already, so there's no need to do anything.
        return new_fc
    # Copy the tutorial dataset's TractCentroids feature class to the new feature class
    print(f"Creating {new_fc} for test input...")
    orig_fc = os.path.join(sf_gdb, "Analysis", "TractCentroids")
    if not arcpy.Exists(orig_fc):
        raise ValueError(f"{orig_fc} is missing.")
    arcpy.management.Copy(orig_fc, new_fc)
    # Add and populate the StoreID field
    # Also add a pre-populated CurbApproach field to test field transfer
    arcpy.management.AddField(new_fc, "StoreID", "TEXT", field_length=8)
    arcpy.management.AddField(new_fc, "CurbApproach", "SHORT")
    store_ids = [  # Pre-assigned store IDs to add to TractCentroids
        'Store_6', 'Store_6', 'Store_11', 'Store_11', 'Store_11', 'BadStore', 'Store_11', 'Store_11', 'Store_11', '',
        'Store_11', 'Store_11', 'Store_6', 'Store_11', 'Store_11', 'Store_11', 'Store_11', 'Store_1', 'Store_7',
        'Store_1', 'Store_1', 'Store_2', 'Store_2', 'Store_2', 'Store_1', 'Store_2', 'Store_7', 'Store_1', 'Store_2',
        'Store_2', 'Store_7', 'Store_7', 'Store_7', 'Store_4', 'Store_4', 'Store_3', 'Store_3', 'Store_3', 'Store_3',
        'Store_19', 'Store_14', 'Store_19', 'Store_19', 'Store_14', 'Store_19', 'Store_16', 'Store_14', 'Store_14',
        'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_14',
        'Store_14', 'Store_7', 'Store_7', 'Store_7', 'Store_7', 'Store_7', 'Store_7', 'Store_7', 'Store_7', 'Store_12',
        'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_7', 'Store_12', 'Store_12',
        'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_12', 'Store_4', 'Store_12',
        'Store_4', 'Store_4', 'Store_4', 'Store_4', 'Store_4', 'Store_4', 'Store_13', 'Store_13', 'Store_13',
        'Store_13', 'Store_5', 'Store_13', 'Store_13', 'Store_5', 'Store_5', 'Store_12', 'Store_12', 'Store_14',
        'Store_14', 'Store_12', 'Store_12', 'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_14', 'Store_12',
        'Store_14', 'Store_12', 'Store_14', 'Store_14', 'Store_14', None, 'Store_12', 'Store_14', 'Store_14',
        'Store_14', 'Store_14', 'Store_3', 'Store_2', 'Store_3', 'Store_3', 'Store_3', 'Store_3', 'Store_4', 'Store_3',
        'Store_2', 'Store_3', 'Store_16', 'Store_3', 'Store_3', 'Store_3', 'Store_3', 'Store_18', 'Store_16',
        'Store_16', 'Store_16', 'Store_15', 'Store_16', 'Store_16', 'Store_14', 'Store_16', 'Store_3', 'Store_3',
        'Store_13', 'Store_3', 'Store_16', 'Store_16', 'Store_16', 'Store_16', 'Store_14', 'Store_14', 'Store_16',
        'Store_16', 'Store_16', 'Store_16', 'Store_16', 'Store_17', 'Store_15', 'Store_17', 'Store_17', 'Store_17',
        'Store_17', 'Store_15', 'Store_15', 'Store_15', 'Store_3', 'Store_15', 'Store_15', 'Store_15', 'Store_15',
        'Store_15', 'Store_15', 'Store_15', 'Store_15', 'Store_15', 'Store_15', 'Store_16', 'Store_19', 'Store_19',
        'Store_19', 'Store_15', 'Store_19', 'Store_15', 'Store_16', 'Store_19', 'Store_19', 'Store_19', 'Store_18',
        'Store_18', 'Store_15', 'Store_18', 'Store_18', 'Store_18', 'Store_15', 'Store_18', 'Store_17', 'Store_18',
        'Store_15', 'Store_15', 'Store_16', 'Store_19', 'Store_15']
    with arcpy.da.UpdateCursor(new_fc, ["StoreID", "CurbApproach"]) as cur:  # pylint: disable=no-member
        idx = 0
        for _ in cur:
            cur.updateRow([store_ids[idx], 2])
            idx += 1
    return new_fc
