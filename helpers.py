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
import traceback
import arcpy

arcgis_version = arcpy.GetInstallInfo()["Version"]

# Set some shared global variables that can be referenced from the other scripts
ID_FIELD_TYPES = ["Short", "Long", "Double", "Single", "Text", "OID"]
MSG_STR_SPLITTER = " | "
DISTANCE_UNITS = ["Kilometers", "Meters", "Miles", "Yards", "Feet", "NauticalMiles"]
TIME_UNITS = ["Days", "Hours", "Minutes", "Seconds"]
OUTPUT_FORMATS = ["Feature class", "CSV files"]
PAIR_TYPES = [
    "A field in Origins defines the assigned Destination (one-to-one)",
    "A separate table defines the origin-destination pairs (many-to-many)"
]
if arcgis_version >= "2.9":
    # The ODCostMatrix solver object's toArrowTable method was added at Pro 2.9. Allow this output format only
    # in software versions that support it.
    OUTPUT_FORMATS.append("Apache Arrow files")
MAX_AGOL_PROCESSES = 4  # AGOL concurrent processes are limited so as not to overload the service for other users.
DATETIME_FORMAT = "%Y%m%d %H:%M"  # Used for converting between datetime and string

# Conversion between ArcGIS field types and python types for use when creating dataframes
PD_FIELD_TYPES = {"String": str, "Single": float, "Double": float, "SmallInteger": int, "Integer": int, "OID": int}


def is_nds_service(network_data_source):
    """Determine if the network data source points to a service.

    Args:
        network_data_source (network data source): Network data source to check.

    Returns:
        bool: True if the network data source is a service URL. False otherwise.
    """
    if not isinstance(network_data_source, str):
        # Probably a network dataset layer
        return False
    return bool(network_data_source.startswith("http"))


def get_tool_limits_and_is_agol(network_data_source, service_name, tool_name):
    """Retrieve a dictionary of various limits supported by a portal tool and whether the portal uses AGOL services.

    Assumes that we have already determined that the network data source is a service.

    Args:
        network_data_source (str): URL to the service being used as the network data source.
        service_name (str): Name of the service, such as "asyncODCostMatrix" or "asyncRoute".
        tool_name (_type_): Tool name for the designated service, such as "GenerateOriginDestinationCostMatrix" or
            "FindRoutes".

    Returns:
        (dict, bool): Dictionary of service limits; Boolean indicating if the service is ArcGIS Online or a hybrid
            portal that falls back to ArcGIS Online.
    """
    arcpy.AddMessage("Getting tool limits from the portal...")
    try:
        tool_info = arcpy.nax.GetWebToolInfo(service_name, tool_name, network_data_source)
        # serviceLimits returns the maximum origins and destinations allowed by the service, among other things
        service_limits = tool_info["serviceLimits"]
        # isPortal returns True for Enterprise portals and False for AGOL or hybrid portals that fall back to using
        # the AGOL services
        is_agol = not tool_info["isPortal"]
        return service_limits, is_agol
    except Exception:
        arcpy.AddError("Error getting tool limits from the portal.")
        errs = traceback.format_exc().splitlines()
        for err in errs:
            arcpy.AddError(err)
        raise


def update_agol_max_processes(max_processes):
    """Update the maximum allowed parallel processes for AGOL if necessary.

    Args:
        max_processes (int): User's desired max parallel processes

    Returns:
        int: Updated max processes <= max allowed for AGOL.
    """
    if max_processes > MAX_AGOL_PROCESSES:
        arcpy.AddWarning((
            f"The specified maximum number of parallel processes, {max_processes}, exceeds the limit of "
            f"{MAX_AGOL_PROCESSES} allowed when using as the network data source the ArcGIS Online "
            "services or a hybrid portal whose network analysis services fall back to the ArcGIS Online "
            "services. The maximum number of parallel processes has been reduced to "
            f"{MAX_AGOL_PROCESSES}."))
        max_processes = MAX_AGOL_PROCESSES
    return max_processes


def convert_time_units_str_to_enum(time_units):
    """Convert a string representation of time units to an arcpy.nax enum.

    Args:
        time_units (str): String representation of time units

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.TimeUnits enum value.

    Returns:
        arcpy.nax.TimeUnits: Time units enum for use in arcpy.nax solver objects
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

    Args:
        distance_units (str): String representation of distance units

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.DistanceUnits enum value.

    Returns:
        arcpy.nax.DistanceUnits: Distance units enum for use in arcpy.nax solver objects
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


def convert_output_format_str_to_enum(output_format) -> OutputFormat:
    """Convert a string representation of the desired output format to an enum.

    Args:
        output_format (str): String representation of the output format

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.DistanceUnits enum value.

    Returns:
        OutputFormat: Output format enum value
    """
    if output_format.lower() == "feature class":
        return OutputFormat.featureclass
    if output_format.lower() == "csv files":
        return OutputFormat.csv
    if output_format.lower() == "apache arrow files":
        return OutputFormat.arrow
    # If we got to this point, the output format was invalid.
    err = f"Invalid output format: {output_format}"
    arcpy.AddError(err)
    raise ValueError(err)


class PreassignedODPairType(enum.Enum):
    """Enum definining the type of preassigned OD pairs being used in the analysis."""

    one_to_one = 1  # Each origin is assigned to exactly one destination.
    many_to_many = 2  # Origins and destinations may be reused. A separate table defines OD pairs.


def convert_pair_type_str_to_enum(pair_type):
    """Convert a string representation of the OD pair assignment type to an enum.

    Args:
        output_format (str): String representation of the output format

    Raises:
        ValueError: If the string cannot be parsed as a valid arcpy.nax.DistanceUnits enum value.

    Returns:
        OutputFormat: Output format enum value
    """
    if "one-to-one" in pair_type:
        return PreassignedODPairType.one_to_one
    if "many-to-many" in pair_type:
        return PreassignedODPairType.many_to_many
    # If we got to this point, the input OD pair assignment type was invalid.
    err = f"Invalid OD pair assignment type: {pair_type}"
    arcpy.AddError(err)
    raise ValueError(err)


def validate_input_feature_class(feature_class):
    """Validate that the designated input feature class exists and is not empty.

    Args:
        feature_class (str, layer): Input feature class or layer to validate

    Raises:
        ValueError: The input feature class does not exist.
        ValueError: The input feature class has no rows.
    """
    if not arcpy.Exists(feature_class):
        err = f"Input dataset {feature_class} does not exist."
        arcpy.AddError(err)
        raise ValueError(err)
    if int(arcpy.management.GetCount(feature_class).getOutput(0)) <= 0:
        err = f"Input dataset {feature_class} has no rows."
        arcpy.AddError(err)
        raise ValueError(err)


def validate_network_data_source(network_data_source):
    """Validate the network data source and return its string-based representation.

    Check out the Network Analyst extension license if relevant.

    Args:
        network_data_source: The network data source from the tool inputs.

    Raises:
        ValueError: If the network dataset doesn't exist
        RuntimeError: If the Network Analyst extension can't be checked out.

    Returns:
        str: Network data source URL or catalog path suitable for passing as a command line argument.
    """
    is_service = is_nds_service(network_data_source)
    if not is_service and not arcpy.Exists(network_data_source):
        err = f"Input network dataset {network_data_source} does not exist."
        arcpy.AddError(err)
        raise ValueError(err)
    if is_service:
        # Add a trailing slash to the URL if needed to avoid potential problems later
        if not network_data_source.endswith("/"):
            network_data_source = network_data_source + "/"
    else:
        # Try to check out the Network Analyst extension
        try:
            arcpy.CheckOutExtension("network")
        except Exception as ex:
            err = "Unable to check out Network Analyst extension license."
            arcpy.AddError(err)
            raise RuntimeError(err) from ex
        # If the network dataset is a layer, convert it to a catalog path so we can pass it to the subprocess
        if hasattr(network_data_source, "dataSource"):
            network_data_source = network_data_source.dataSource
    return network_data_source


def precalculate_network_locations(input_features, network_data_source, travel_mode, config_file_props):
    """Precalculate network location fields if possible for faster loading and solving later.

    Cannot be used if the network data source is a service. Uses the searchTolerance, searchToleranceUnits, and
    searchQuery properties set in the config file.

    Args:
        input_features (feature class catalog path): Feature class to calculate network locations for
        network_data_source (network dataset catalog path): Network dataset to use to calculate locations
        travel_mode (travel mode): Travel mode name, object, or json representation to use when calculating locations.
        config_file_props (dict): Dictionary of solver object properties from config file.
    """
    arcpy.AddMessage(f"Precalculating network location fields for {input_features}...")

    # Get location settings from config file if present
    search_tolerance = None
    if "searchTolerance" in config_file_props and "searchToleranceUnits" in config_file_props:
        search_tolerance = f"{config_file_props['searchTolerance']} {config_file_props['searchToleranceUnits'].name}"
    search_query = config_file_props.get("search_query", None)

    # Calculate network location fields if network data source is local
    arcpy.na.CalculateLocations(
        input_features, network_data_source,
        search_tolerance=search_tolerance,
        search_query=search_query,
        travel_mode=travel_mode
    )


def get_oid_ranges_for_input(input_fc, max_chunk_size):
    """Construct ranges of ObjectIDs for use in where clauses to split large data into chunks.

    Args:
        input_fc (str, layer): Data that needs to be split into chunks
        max_chunk_size (int): Maximum number of rows that can be in a chunk

    Returns:
        list: list of ObjectID ranges for the current dataset representing each chunk. For example,
            [[1, 1000], [1001, 2000], [2001, 2478]] represents three chunks of no more than 1000 rows.
    """
    ranges = []
    num_in_range = 0
    current_range = [0, 0]
    # Loop through all OIDs of the input and construct tuples of min and max OID for each chunk
    # We do it this way and not by straight-up looking at the numerical values of OIDs to account
    # for definition queries, selection sets, or feature layers with gaps in OIDs
    for row in arcpy.da.SearchCursor(input_fc, "OID@"):  # pylint: disable=no-member
        oid = row[0]
        if num_in_range == 0:
            # Starting new range
            current_range[0] = oid
        # Increase the count of items in this range and set the top end of the range to the current oid
        num_in_range += 1
        current_range[1] = oid
        if num_in_range == max_chunk_size:
            # Finishing up a chunk
            ranges.append(current_range)
            # Reset range trackers
            num_in_range = 0
            current_range = [0, 0]
    # After looping, close out the last range if we still have one open
    if current_range != [0, 0]:
        ranges.append(current_range)

    return ranges


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


def run_gp_tool(log_to_use, tool, tool_args=None, tool_kwargs=None):
    """Run a geoprocessing tool with nice logging.

    The purpose of this function is simply to wrap the call to a geoprocessing tool in a way that we can log errors,
    warnings, and info messages as well as tool run time into our logging. This helps pipe the messages back to our
    script tool dialog.

    Args:
        tool (arcpy geoprocessing tool class): GP tool class command, like arcpy.management.CreateFileGDB
        tool_args (list, optional): Ordered list of values to use as tool arguments. Defaults to None.
        tool_kwargs (dictionary, optional): Dictionary of tool parameter names and values that can be used as named
            arguments in the tool command. Defaults to None.
        log_to_use (logging.logger, optional): logger class to use for messages. Defaults to LOGGER. When calling this
            from the Route class, use self.logger instead so the messages go to the processes's log file instead
            of stdout.

    Returns:
        GP result object: GP result object returned from the tool run.

    Raises:
        arcpy.ExecuteError if the tool fails
    """
    # Try to retrieve and log the name of the tool
    tool_name = repr(tool)
    try:
        tool_name = tool.__esri_toolname__
    except Exception:  # pylint: disable=broad-except
        try:
            tool_name = tool.__name__
        except Exception:  # pylint: disable=broad-except
            # Probably the tool didn't have an __esri_toolname__ property or __name__. Just don't worry about it.
            pass
    log_to_use.debug(f"Running geoprocessing tool {tool_name}...")

    # Try running the tool, and log all messages
    try:
        if tool_args is None:
            tool_args = []
        if tool_kwargs is None:
            tool_kwargs = {}
        result = tool(*tool_args, **tool_kwargs)
        info_msgs = [msg for msg in result.getMessages(0).splitlines() if msg]
        warning_msgs = [msg for msg in result.getMessages(1).splitlines() if msg]
        for msg in info_msgs:
            log_to_use.debug(msg)
        for msg in warning_msgs:
            log_to_use.warning(msg)
    except arcpy.ExecuteError:
        log_to_use.error(f"Error running geoprocessing tool {tool_name}.")
        # First check if it's a tool error and if so, handle warning and error messages.
        info_msgs = [msg for msg in arcpy.GetMessages(0).strip("\n").splitlines() if msg]
        warning_msgs = [msg for msg in arcpy.GetMessages(1).strip("\n").splitlines() if msg]
        error_msgs = [msg for msg in arcpy.GetMessages(2).strip("\n").splitlines() if msg]
        for msg in info_msgs:
            log_to_use.debug(msg)
        for msg in warning_msgs:
            log_to_use.warning(msg)
        for msg in error_msgs:
            log_to_use.error(msg)
        raise
    except Exception:
        # Unknown non-tool error
        log_to_use.error(f"Error running geoprocessing tool {tool_name}.")
        errs = traceback.format_exc().splitlines()
        for err in errs:
            log_to_use.error(err)
        raise

    log_to_use.debug(f"Finished running geoprocessing tool {tool_name}.")
    return result
