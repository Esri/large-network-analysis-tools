"""Helper functions for large network analysis tools.

This is a sample script users can modify to fit their specific needs.

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
import arcpy


def get_catalog_path(data):
    """Get the catalog path for the designated input if possible.

    Ensures we can pass map layers to the subprocess. This works if the input is a layer object. It will not work if the
    input is a layer name, so do not use it for that.

    Args:
        data (layer or str): Layer from which to retrieve the catalog path.

    Returns:
        string: Catalog path to the data
    """
    if hasattr(data, "dataSource"):
        return data.dataSource
    else:
        # Assume it is already the catalog path. Will not work if it's a layer name.
        return data


def is_nds_service(network_data_source):
    """Determine if the network data source points to a service.

    Args:
        network_data_source (network data source): Network data source to check.

    Returns:
        bool: True if the network data source is a service URL. False otherwise.
    """
    return True if network_data_source.startswith("http") else False


def convert_time_units_str_to_enum(time_units):
    """Convert a string representation of time units to an arcpy.nax enum.

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.TimeUnits enum value.
    """
    if time_units.lower() == "minutes":
        return arcpy.nax.TimeUnits.Minutes
    elif time_units.lower() == "seconds":
        return arcpy.nax.TimeUnits.Seconds
    elif time_units.lower() == "hours":
        return arcpy.nax.TimeUnits.Hours
    elif time_units.lower() == "days":
        return arcpy.nax.TimeUnits.Days
    else:
        # If we got to this point, the input time units were invalid.
        err = f"Invalid time units: {time_units}"
        arcpy.AddError(err)
        raise ValueError(err)


def convert_distance_units_str_to_enum(distance_units):
    """Convert a string representation of distance units to an arcpy.nax.DistanceUnits enum.

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.DistanceUnits enum value.
    """
    if distance_units.lower() == "miles":
        return arcpy.nax.DistanceUnits.Miles
    elif distance_units.lower() == "kilometers":
        return arcpy.nax.DistanceUnits.Kilometers
    elif distance_units.lower() == "meters":
        return arcpy.nax.DistanceUnits.Meters
    elif distance_units.lower() == "feet":
        return arcpy.nax.DistanceUnits.Feet
    elif distance_units.lower() == "yards":
        return arcpy.nax.DistanceUnits.Yards
    elif distance_units.lower() == "nauticalmiles" or distance_units.lower() == "nautical miles":
        return arcpy.nax.DistanceUnits.NauticalMiles
    else:
        # If we got to this point, the input distance units were invalid.
        err = f"Invalid distance units: {distance_units}"
        arcpy.AddError(err)
        raise ValueError(err)
