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

from ParallelODCM import MSG_STR_SPLITTER


def is_nds_service(network_data_source):
    """Determine if the network data source points to a service.

    Args:
        network_data_source (network data source): Network data source to check.

    Returns:
        bool: True if the network data source is a service URL. False otherwise.
    """
    return True if network_data_source.startswith("http") else False


def get_travel_mode_string(travel_mode):
    """Get a string representation of a travel mode if possible.

    Args:
        travel_mode (arcpy.nax.TravelMode, str): Travel mode to convert to a string

    Raises:
        ValueError: The travel mode is invalid

    Returns:
        str: JSON string representation of the travel mode or the travel mode's name
    """
    if isinstance(travel_mode, str):
        # The travel mode is already a string. It's either a string name or a JSON string representation of the travel
        # mode. Just return it as is.
        return travel_mode
    if isinstance(travel_mode, arcpy.nax.TravelMode):
        if hasattr(travel_mode, "_JSON"):
            return travel_mode._JSON  # pylint: disable=protected-access
        else:
            return travel_mode.name
    # If we got to this point, the travel mode is invalid.
    err = f"Invalid travel mode: {travel_mode}"
    arcpy.AddError(err)
    raise ValueError(err)


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


def parse_std_and_write_to_gp_ui(msg_string):
    """Parse a message string returned from the subprocess's stdout and write it to the GP UI according to type.

    Logged messages in the ParallelODCM module start with a level indicator that allows us to parse them and write them
    as errors, warnings, or info messages.  Example: "ERROR | Something terrible happened" is an error message.

    Args:
        msg_string (str): Message string (already decoded) returned from ParallelODCM.py subprocess stdout
    """
    try:
        level, msg = msg_string.split(MSG_STR_SPLITTER)
        if level in ["ERROR", "CRITICAL"]:
            arcpy.AddError(msg)
        elif level == "WARNING":
            arcpy.AddWarning(msg)
        else:
            arcpy.AddMessage(msg)
    except Exception:  # pylint: disable=broad-except
        arcpy.AddMessage(msg_string)
