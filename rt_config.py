"""Defines Route solver object properties that are not specified
in the tool dialog.

A list of Route solver properties is documented here:
https://pro.arcgis.com/en/pro-app/latest/arcpy/network-analyst/route.htm

You can include any of them in the dictionary in this file, and the tool will
use them. However, travelMode, timeUnits, distanceUnits, defaultImpedanceCutoff,
and defaultDestinationCount will be ignored because they are specified in the
tool dialog.  TODO

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
import arcpy

# These properties are set by the tool dialog or can be specified as command line arguments. Do not set the values for
# these properties in the RT_PROPS dictionary below because they will be ignored.
RT_PROPS_SET_BY_TOOL = [
    # TODO
    "travelMode", "timeUnits", "distanceUnits", "timeOfDay"]

# You can customize these properties to your needs, and the parallel Route calculations will use them.
RT_PROPS = {
    'accumulateAttributeNames': [],
    'allowSaveLayerFile': False,
    'allowSaveRouteData': False,
    'directionsDistanceUnits': arcpy.nax.DistanceUnits.Kilometers,
    'directionsLanguage': "en",
    'directionsStyle': arcpy.nax.DirectionsStyle.Desktop,
    'findBestSequence': False,
    'ignoreInvalidLocations': True,
    'overrides': "",
    'preserveFirstStop': False,
    'preserveLastStop': False,
    'returnDirections': False,
    'returnRouteEdges': True,
    'returnRouteJunctions': False,
    'returnRouteTurns': False,
    'returnToStart': False,
    'routeShapeType': arcpy.nax.RouteShapeType.TrueShapeWithMeasures,
    # 'searchQuery': "",  # This parameter is very network specific. Only uncomment if you are using it.
    'searchTolerance': 5000,
    'searchToleranceUnits': arcpy.nax.DistanceUnits.Meters,
    'timeZone': arcpy.nax.TimeZoneUsage.LocalTimeAtLocations,
    'timeZoneForTimeWindows': arcpy.nax.TimeZoneUsage.LocalTimeAtLocations,
}
