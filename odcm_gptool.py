"""Execution code for Solve Origin Destination Cost Matrix tool."""
################################################################################
'''Copyright 2020 Esri
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

import os
import sys
import subprocess
import arcpy


def solve_odcm():
    """Read tool inputs and call the origin destination cost matrix class that performs the computation."""
    # Read tool inputs
    origins = arcpy.GetParameter(0)
    destinations = arcpy.GetParameter(1)
    network_data_source = arcpy.GetParameter(2)
    travel_mode = arcpy.GetParameterAsText(3)
    output_folder = arcpy.GetParameterAsText(4)
    cutoff = arcpy.GetParameterAsText(5)
    if not cutoff:
        cutoff = "0"
    target_count = arcpy.GetParameterAsText(6)
    max_od_size = arcpy.GetParameterAsText(7)
    num_workers = arcpy.GetParameterAsText(8)

    # Get catalog paths to origins feature class, destinations feature class and the network data source since the ODCM
    # tool requires catalog paths.
    # arcpy.GetParameter returns a layer object if a layer is specified as input to the GP tool. The layer object
    # has a dataSource property that contains the catalog path.
    # arcpy.GetParameterAsText returns a GP value object whose value property can be used to get the value as string.
    if hasattr(origins, "dataSource"):
        origins = origins.dataSource
    else:
        origins = origins.value
    if hasattr(destinations, "dataSource"):
        destinations = destinations.dataSource
    else:
        destinations = destinations.value
    if hasattr(network_data_source, "dataSource"):
        network_data_source = network_data_source.dataSource
    else:
        network_data_source = network_data_source.value

    # The ODCM command line tool requires a value of zero to be passed as target count to find all destinations
    if not target_count:
        target_count = "0"

    # arcpy.AddMessage(origins)
    # arcpy.AddMessage(destinations)
    # arcpy.AddMessage(f"Network data source: {network_data_source}")
    # arcpy.AddMessage(f"cutoff: {cutoff}")
    # arcpy.AddMessage(f"number of destinations to find: {target_count}")

    # Get the directory containing the script
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Create a list of arguments to call the ODCM command line tool.
    odcm_inputs = [os.path.join(sys.exec_prefix, "python.exe"),
                   os.path.join(cwd, "odcm.py"),
                   "--origins", origins,
                   "--destinations", destinations,
                   "--network-data-source", network_data_source,
                   "--travel-mode", travel_mode,
                   "--max-od-size", max_od_size,
                   "--cutoff", cutoff,
                   "--target-count", target_count,
                   "--workers", num_workers,
                   "--folder", output_folder]
    # We do not want to show the console window when calling the command line tool from within our GP tool. This can be
    # done by setting this hex code.
    create_no_window = 0x08000000
    # Store any output messages (failure as well as success) from the command line tool in a log file
    output_msg_file = os.path.join(cwd, "odcm_outputs.txt")
    with open(output_msg_file, "w") as output_fp:
        try:
            odcm_result = subprocess.run(odcm_inputs, stderr=subprocess.STDOUT, stdout=output_fp, check=True,
                                         creationflags=create_no_window)
        except subprocess.CalledProcessError as ex:
            arcpy.AddError(f"Call to ODCM command line tool failed. Check {output_msg_file} for additional details.")
            arcpy.AddError(f"{ex}")
            raise SystemExit(-1)
        # arcpy.AddMessage(output_fp.readlines())


if __name__ == "__main__":
    solve_odcm()
