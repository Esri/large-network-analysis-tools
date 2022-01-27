"""Helper functions for large network analysis tools.

This is a sample script users can modify to fit their specific needs.

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
import enum
import arcpy

arcgis_version = arcpy.GetInstallInfo()["Version"]

# Set some shared global variables that can be referenced from the other scripts
MSG_STR_SPLITTER = " | "
DISTANCE_UNITS = ["Kilometers", "Meters", "Miles", "Yards", "Feet", "NauticalMiles"]
TIME_UNITS = ["Days", "Hours", "Minutes", "Seconds"]
OUTPUT_FORMATS = ["Feature class", "CSV files", "Apache Arrow files"]
MAX_AGOL_PROCESSES = 4  # AGOL concurrent processes are limited so as not to overload the service for other users.


def is_nds_service(network_data_source):
    """Determine if the network data source points to a service.

    Args:
        network_data_source (network data source): Network data source to check.

    Returns:
        bool: True if the network data source is a service URL. False otherwise.
    """
    return bool(network_data_source.startswith("http"))


def convert_time_units_str_to_enum(time_units):
    """Convert a string representation of time units to an arcpy.nax enum.

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.TimeUnits enum value.
    """
    if time_units.lower() == "minutes":
        return arcpy.nax.TimeUnits.Minutes
    if time_units.lower() == "seconds":
        return arcpy.nax.TimeUnits.Seconds
    if time_units.lower() == "hours":
        return arcpy.nax.TimeUnits.Hours
    if time_units.lower() == "days":
        return arcpy.nax.TimeUnits.Days
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
    if distance_units.lower() == "kilometers":
        return arcpy.nax.DistanceUnits.Kilometers
    if distance_units.lower() == "meters":
        return arcpy.nax.DistanceUnits.Meters
    if distance_units.lower() == "feet":
        return arcpy.nax.DistanceUnits.Feet
    if distance_units.lower() == "yards":
        return arcpy.nax.DistanceUnits.Yards
    if distance_units.lower() == "nauticalmiles" or distance_units.lower() == "nautical miles":
        return arcpy.nax.DistanceUnits.NauticalMiles
    # If we got to this point, the input distance units were invalid.
    err = f"Invalid distance units: {distance_units}"
    arcpy.AddError(err)
    raise ValueError(err)


class OutputFormat(enum.Enum):
    """Enum defining the output format for the OD Cost Matrix results."""

    featureclass = 1
    csv = 2
    arrow = 3


def convert_output_format_str_to_enum(output_format):
    """Convert a string representation of the desired output format to an enum.

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.DistanceUnits enum value.
    """
    if output_format.lower() == "feature class":
        return OutputFormat.featureclass
    if output_format.lower() == "csv files":
        return OutputFormat.csv
    if output_format.lower() == "apache arrow files":
        return OutputFormat.arrow
    # If we got to this point, the input distance units were invalid.
    err = f"Invalid output format: {output_format}"
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
