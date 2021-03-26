
# large-network-analysis-tools

The tools and code samples here help you solve large network analysis problems in ArcGIS Pro. We have provided a python script that can solve a large origin destination cost matrix problem by chunking the input data, solving in parallel, and combining the results into a single output.

## Features
* The LargeNetworkAnalysisTools.pyt toolbox has a geoprocessing tool called "Solve Large OD Cost Matrix" that can be used to solve large origin destination cost matrix problems. You can run this tool as-is out of the box with no changes to the code.
* The odcm.py script does all the work. You can modify this script to suit your needs, or you can use it as an example when writing your own script.

## Instructions

1. Download the latest release
2. Modify the code to suit your needs
3. Run the code in standalone python, or run the provided geoprocessing tool from within ArcGIS Pro.

### Solve Large OD Cost Matrix tool inputs
- **Origins** - The feature class or layer containing the origins
- **Destinations** - The feature class or layer containing the destinations
- **Output OD Lines Feature Class** - Path to the output feature class that will contain the OD Cost Matrix Lines output computed by the tool. The schema of this feature class is described in the [arcpy documentation](https://pro.arcgis.com/en/pro-app/latest/arcpy/network-analyst/origindestinationcostmatrix-output-data-types.htm#ESRI_SECTION1_9FF9489173C741DD95472F21B5AD8374).
- **Output Updated Origins** - Path to the output feature class that will contain the updated origins, which may be spatially sorted and have added fields. The OriginOID field in the Output OD Lines Feature Class refers to the ObjectID of the Output Updated Origins and not the original input origins.
- **Output Updated Destinations** - Path to the output feature class that will contain the updated destinations, which may be spatially sorted and have added fields. The DestinationOID field in the Output OD Lines Feature Class refers to the ObjectID of the Output Updated Destinations and not the original input destinations.
- **Network Data Source** - Network dataset, network dataset layer, or portal URL to use when calculating the OD Cost Matrix.
- **Travel Mode** - Network travel mode to use when calculating the OD Cost Matrix
- **Time Units** - The time units the output Total_Time field will be reported in.
- **Distance Units** - The distance units the output Total_Distance field will be reported in.
- **Maximum Origins and Destinations per Chunk** - Defines the chunk size for parallel OD Cost Matrix calculations. For example, if you want to process a maximum of 1000 origins and 1000 destinations in a single chunk, set this parameter to 1000.
- **Maximum Number of Parallel Processes** - Defines the maximum number of parallel processes to run at once. Do not exceed the number of cores of your machine.
- **Cutoff** - Impedance cutoff limiting the search distance for each origin. For example, you could set up the problem to find only destinations within a 15 minute drive time of the origins. This parameter is optional. Leaving it blank uses no cutoff.
  - If your travel mode has time-based impedance units, Cutoff represents a time and is interpreted in the units specified in the Time Units parameter.
  - If your travel mode has distance-based impedance units, Cutoff represents a distance and is interpreted in the units specified in the Distance Units parameter.
  - If your travel mode has other units (not time- or distance-based), Cutoff should be specified in the units of your travel mode's impedance attribute.
- **Number of Destinations to Find for Each Origin** - The number of destinations to find for each origin. For example, setting this to 3 will result in the output including the travel time and distance from each origin to its three closest destinations. This parameter is optional. Leaving it blank results in finding the travel time and distance from each origin to all destinations.
- **Barriers** - Point, line, or polygon barriers to use in the OD Cost Matrix analysis. This parameter is optional.
- **Precalculate Network Locations** - When you solve a network analysis, the input points must "locate" on the network used for the analysis. When chunking your inputs to solve in parallel, inputs may be used many times. Rather than calculating the network location fields for each input every time it is used, it is more efficient to calculate all the network location fields up front and re-use them. Set this parameter to True to pre-calculate the network location fields. This is recommended for every situation unless:
  - You are using a portal URL as the network data source. In this case, pre-calculating network locations is not possible, and the parameter is hidden.
  - You have already pre-calculated the network location fields using the network dataset and travel mode you are using for this analysis. In this case, you can save time by not precalculating them again.


## Requirements

* ArcGIS Pro 2.5 or later
* One of the following three options:
  * A routable [network dataset](https://pro.arcgis.com/en/pro-app/help/analysis/networks/what-is-network-dataset-.htm) and the Network Analyst extension license
  * An ArcGIS Online account with routing privileges and sufficient [credits](https://pro.arcgis.com/en/pro-app/tool-reference/appendices/geoprocessing-tools-that-use-credits.htm#ESRI_SECTION1_3EF40A7C01C042D8A76DB9518B793E9E)
  * A portal with [ArcGIS Enterprise routing services](https://pro.arcgis.com/en/pro-app/help/analysis/networks/using-arcgis-enterprise-routing-services.htm) configured.
* Origin and destination points you wish to analyze

## Resources

* [OD Cost Matrix tutorial](https://pro.arcgis.com/en/pro-app/help/analysis/networks/od-cost-matrix-tutorial.htm)
* [Network Analyst arcpy.nax python module documentation](https://pro.arcgis.com/en/pro-app/arcpy/network-analyst/what-is-the-network-analyst-module.htm)
* [Video presentation about solving large problems from DevSummit 2020](https://youtu.be/9PI7HIm1y8U)

## Issues

Find a bug or want to request a new feature?  Please let us know by submitting an issue.

## Contributing

Esri welcomes contributions from anyone and everyone. Please see our [guidelines for contributing](https://github.com/esri/contributing).

## Licensing
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

A copy of the license is available in the repository's [license.txt](license.txt) file.