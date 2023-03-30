"""Helper for unit tests to create required inputs.

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
import os
import csv
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


def get_od_pair_csv(input_data_folder):
    """Create the od_pairs.csv input file in the input data folder for use in unit testing."""
    od_pair_file = os.path.join(input_data_folder, "od_pairs.csv")
    if os.path.exists(od_pair_file):
        # The OD pair file already exists, so no need to do anything.
        return od_pair_file
    print(f"Creating {od_pair_file} for test input...")
    od_pairs = [
        ["06075011400", "Store_13"],
        ["06075011400", "Store_19"],
        ["06075011400", "Store_11"],
        ["06075021800", "Store_9"],
        ["06075021800", "Store_12"],
        ["06075013000", "Store_3"],
        ["06075013000", "Store_10"],
        ["06075013000", "Store_1"],
        ["06075013000", "Store_8"],
        ["06075013000", "Store_12"],
        ["06081602500", "Store_12"],
        ["06081602500", "Store_25"],
        ["06081602500", "Store_9"],
        ["06081602500", "Store_17"],
        ["06081602500", "Store_21"],
        ["06075030400", "Store_7"],
        ["06075030400", "Store_5"],
        ["06075030400", "Store_21"],
        ["06075030400", "Store_19"],
        ["06075030400", "Store_23"],
        ["06075045200", "Store_1"],
        ["06075045200", "Store_5"],
        ["06075045200", "Store_6"],
        ["06075012600", "Store_19"],
        ["06075012600", "Store_5"],
        ["06075012600", "Store_23"],
        ["06075012600", "Store_15"],
        ["06075060700", "Store_19"],
        ["06081601400", "Store_7"],
        ["06081601400", "Store_15"],
        ["06081601400", "Store_2"],
        ["06075023001", "Store_22"],
        ["06075032800", "Store_25"],
        ["06081601800", "Store_13"],
        ["06075013100", "Store_25"],
        ["06075013100", "Store_9"],
        ["06075013100", "Store_23"],
        ["06081600600", "Store_22"],
        ["06081600600", "Store_12"],
        ["06081600600", "Store_1"],
        ["06081600600", "Store_21"],
        ["06075012000", "Store_22"],
        ["06075031400", "Store_20"],
        ["06075031400", "Store_24"],
        ["06081601000", "Store_2"],
        ["06075026004", "Store_16"],
        ["06075026004", "Store_7"],
        ["06075020300", "Store_17"],
        ["06075020300", "Store_13"],
        ["06075060502", "Store_18"],
        ["06075011000", "Store_21"],
        ["06075011000", "Store_19"],
        ["06075011000", "Store_12"],
        ["06075011000", "Store_22"],
        ["06075026404", "Store_17"],
        ["06075026404", "Store_9"],
        ["06081601601", "Store_9"],
        ["06075021500", "Store_22"],
        ["06075021500", "Store_9"],
        ["06075026302", "Store_21"],
        ["06075026302", "Store_15"],
        ["06075026302", "Store_23"],
        ["06075026302", "Store_24"]
    ]
    with open(od_pair_file, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(od_pairs)
    return od_pair_file


def get_od_pairs_fgdb_table(sf_gdb):
    """Create the ODPairs fgdb input table in SanFrancisco.gdb for use in unit testing."""
    od_pair_table = os.path.join(sf_gdb, "ODPairs")
    if arcpy.Exists(od_pair_table):
        # The OD pair table already exists, so no need to do anything.
        return od_pair_table
    input_data_folder = os.path.dirname(sf_gdb)
    od_pair_csv = get_od_pair_csv(input_data_folder)
    print(f"Creating {od_pair_table} for test input...")
    arcpy.management.CreateTable(sf_gdb, "ODPairs")
    arcpy.management.AddFields(
        od_pair_table,
        [["OriginID", "TEXT", "OriginID", 25], ["DestinationID", "TEXT", "DestinationID", 45]]
    )
    with arcpy.da.InsertCursor(od_pair_table, ["OriginID", "DestinationID"]) as cur:  # pylint: disable=no-member
        with open(od_pair_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                cur.insertRow(row)

    return od_pair_table
