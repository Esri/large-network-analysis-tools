
# large-network-analysis-tools

The tools and code samples here help you solve large network analysis problems in ArcGIS Pro. We have provided a python script that can solve a large origin destination cost matrix problem by chunking the input data, solving in parallel, and combining the results into a single output.

## Features
* The LargeNetworkAnalysisTools.pyt toolbox has a geoprocessing tool called "Solve Large OD Cost Matrix" that can be used to solve large origin destination cost matrix problems. You can run this tool as-is out of the box with no changes to the code.
* The odcm.py script does all the work. You can modify this script to suit your needs, or you can use it as an example when writing your own script.

## Instructions

1. Download the latest release
2. Modify the code to suit your needs
3. Run the code in standalone python, or run the provided geoprocessing tool from within ArcGIS Pro.

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