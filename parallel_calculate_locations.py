"""Calculate the network locations for a large dataset by chunking the
inputs and solving in parallel.

This is a sample script users can modify to fit their specific needs.

This script is intended to be called as a subprocess from a other scripts
so that it can launch parallel processes with concurrent.futures. It must be
called as a subprocess because the main script tool process, when running
within ArcGIS Pro, cannot launch parallel subprocesses on its own.

This script should not be called directly from the command line.

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
# pylint: disable=logging-fstring-interpolation
from concurrent import futures
import os
import sys
import uuid
import logging
import shutil
import itertools
import time
import datetime
import traceback
import argparse
import csv
import pandas as pd

import arcpy

import helpers

arcpy.env.overwriteOutput = True

# Set logging for the main process.
# LOGGER logs everything from the main process to stdout using a specific format that the calling tool
# can parse and write to the geoprocessing message feed.
LOG_LEVEL = logging.INFO  # Set to logging.DEBUG to see verbose debug messages
LOGGER = logging.getLogger(__name__)  # pylint:disable=invalid-name
LOGGER.setLevel(LOG_LEVEL)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(LOG_LEVEL)
# Used by script tool to split message text from message level to add correct message type to GP window
console_handler.setFormatter(logging.Formatter("%(levelname)s" + helpers.MSG_STR_SPLITTER + "%(message)s"))
LOGGER.addHandler(console_handler)

DELETE_INTERMEDIATE_OUTPUTS = True  # Set to False for debugging purposes


class LocationCalculator(helpers.JobFolderMixin, helpers.LoggingMixin, helpers.MakeNDSLayerMixin):
    """Used for calculating network locations for a designated chunk of the input datasets."""

    def __init__(self, **kwargs):
        """Initialize the OD Cost Matrix analysis for the given inputs.

        Expected arguments:
        - origins
        - destinations
        - output_format
        - output_od_location
        - network_data_source
        - travel_mode
        - time_units
        - distance_units
        - cutoff
        - num_destinations
        - time_of_day
        - scratch_folder
        - barriers
        """
        self.input_fc = kwargs["input_fc"]
        self.network_data_source = kwargs["network_data_source"]
        self.travel_mode = kwargs["travel_mode"]
        self.config_file_props = kwargs["config_file_props"]
        self.scratch_folder = kwargs["scratch_folder"]

        # Create a job ID and a folder for this job
        self._create_job_folder()

        # Setup the class logger. Logs for each parallel process are not written to the console but instead to a
        # process-specific log file.
        self.setup_logger("CalcLocs")

        # Create a network dataset layer if needed
        self._make_nds_layer()

        # Define output feature class path for this chunk (set during feature selection)
        self.out_fc = None

        # Prepare a dictionary to store info about the analysis results
        self.job_result = {
            "jobId": self.job_id,
            "jobFolder": self.job_folder,
            "outputFC": "",
            "logFile": self.log_file
        }

    def _subset_inputs(self, oid_range):
        """Create a layer from the input feature class that contains only the OIDs for this chunk.

        Args:
            oid_range (list): Input feature class ObjectID range to select for this chunk
        """
        # Copy the subset of features in this OID range to a feature class in the job gdb so we can calculate locations
        # on it without interference from other parallel processes
        self.logger.debug("Subsetting features for this chunk...")
        out_gdb = self._create_output_gdb()
        self.out_fc = os.path.join(out_gdb, f"Locs_{oid_range[0]}_{oid_range[1]}")
        oid_field_name = arcpy.Describe(self.input_fc).oidFieldName
        where_clause = (
            f"{oid_field_name} >= {oid_range[0]} "
            f"And {oid_field_name} <= {oid_range[1]}"
        )
        self.logger.debug(f"Where clause: {where_clause}")
        arcpy.conversion.FeatureClassToFeatureClass(
            self.out_fc,
            os.path.dirname(self.out_fc),
            os.path.basename(self.out_fc)
        )

    def calculate_locations(self, oid_range):
        """Calculate locations for a chunk of the input feature class with the designated OID range."""
        self._subset_inputs(oid_range)
        self.logger.debug("Calculating locations...")
        helpers.precalculate_network_locations(
            self.out_fc, self.network_data_source, self.travel_mode, self.config_file_props)
        self.job_result["outputFC"] = self.out_fc


def calculate_locations_for_chunk(calc_locs_settings, chunk):
    location_calculator = LocationCalculator(**calc_locs_settings)
    location_calculator.calculate_locations(chunk)
    location_calculator.teardown_logger()
    return location_calculator.job_result


class ParallelLocationCalculator:
    """Calculates network locations for a large dataset by chunking the dataset and calculating in parallel."""

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self, input_features, network_data_source, travel_mode, chunk_size, max_processes, config_file_props
    ):
        """Compute OD Cost Matrices between Origins and Destinations in parallel and combine results.

        Compute OD cost matrices in parallel and combine and post-process the results.
        This class assumes that the inputs have already been pre-processed and validated.

        Args:
            origins (str): Catalog path to origins
            destinations (str): Catalog path to destinations
            network_data_source (str): Network data source catalog path or URL
            travel_mode (str): String-based representation of a travel mode (name or JSON)
            output_format (str): String representation of the output format
            output_od_location (str): Catalog path to the output feature class or folder where the OD Lines output will
                be stored.
            max_origins (int): Maximum origins allowed in a chunk
            max_destinations (int): Maximum destinations allowed in a chunk
            max_processes (int): Maximum number of parallel processes allowed
            time_units (str): String representation of time units
            distance_units (str): String representation of distance units
            cutoff (float, optional): Impedance cutoff to limit the OD Cost Matrix solve. Interpreted in the time_units
                if the travel mode is time-based. Interpreted in the distance-units if the travel mode is distance-
                based. Interpreted in the impedance units if the travel mode is neither time- nor distance-based.
                Defaults to None. When None, do not use a cutoff.
            num_destinations (int, optional): The number of destinations to find for each origin. Defaults to None,
                which means to find all destinations.
            time_of_day (str): String representation of the start time for the analysis ("%Y%m%d %H:%M" format)
            barriers (list(str), optional): List of catalog paths to point, line, and polygon barriers to use.
                Defaults to None.
        """
        self.input_features = input_features
        self.max_processes = max_processes

        # Scratch folder to store intermediate outputs from the OD Cost Matrix processes
        unique_id = uuid.uuid4().hex
        self.scratch_folder = os.path.join(arcpy.env.scratchFolder, "CalcLocs_" + unique_id)  # pylint: disable=no-member
        LOGGER.info(f"Intermediate outputs will be written to {self.scratch_folder}.")
        os.mkdir(self.scratch_folder)

        # Dictionary of static input settings to send to the parallel location calculator
        self.calc_locs_inputs = {
            "input_fc": self.input_features,
            "network_data_source": network_data_source,
            "travel_mode": travel_mode,
            "config_file_props": config_file_props,
            "scratch_folder": self.scratch_folder
        }

        # List of intermediate output feature classes created by each process
        self.temp_out_fcs = []

        # Construct OID ranges for the input data chunks
        self.ranges = helpers.get_oid_ranges_for_input(self.input_features, chunk_size)

    def calc_locs_in_parallel(self):
        """Calculate locations in parallel."""
        total_jobs = len(self.ranges)
        LOGGER.info(f"Beginning parallel Calculate Locations ({total_jobs} chunks)")
        completed_jobs = 0  # Track the number of jobs completed so far to use in logging
        # Use the concurrent.futures ProcessPoolExecutor to spin up parallel processes that calculate chunks of
        # locations
        with futures.ProcessPoolExecutor(max_workers=self.max_processes) as executor:
            # Each parallel process calls the calculate_locations_for_chunk() function for the given range of OIDs
            # in the input dataset
            jobs = {executor.submit(
                calculate_locations_for_chunk, self.calc_locs_inputs, chunk
                ): chunk for chunk in self.ranges}
            # As each job is completed, add some logging information and store the results to post-process later
            for future in futures.as_completed(jobs):
                try:
                    # The job returns a results dictionary. Retrieve it.
                    result = future.result()
                except Exception:  # pylint: disable=broad-except
                    # If we couldn't retrieve the result, some terrible error happened and the job errored.
                    # Note: This does not mean solve failed. It means some unexpected error was thrown. The most likely
                    # causes are:
                    # a) If you're calling a service, the service was temporarily down.
                    # b) You had a temporary file read/write or resource issue on your machine.
                    # c) If you're actively updating the code, you introduced an error.
                    # To make the tool more robust against temporary glitches, retry submitting the job up to the number
                    # of times designated in helpers.MAX_RETRIES.  If the job is still erroring after that many retries,
                    # fail the entire tool run.
                    errs = traceback.format_exc().splitlines()
                    failed_range = jobs[future]
                    LOGGER.debug((
                        f"Failed to get results for Calculate Locations chunk {failed_range} from the parallel process."
                        f" Will retry up to {helpers.MAX_RETRIES} times. Errors: {errs}"
                    ))
                    job_failed = True
                    num_retries = 0
                    while job_failed and num_retries < helpers.MAX_RETRIES:
                        num_retries += 1
                        try:
                            future = executor.submit(calculate_locations_for_chunk, self.calc_locs_inputs, failed_range)
                            result = future.result()
                            job_failed = False
                            LOGGER.debug(
                                f"Calculate Locations chunk {failed_range} succeeded after {num_retries} retries.")
                        except Exception:  # pylint: disable=broad-except
                            # Update exception info to the latest error
                            errs = traceback.format_exc().splitlines()
                    if job_failed:
                        # The job errored and did not succeed after retries.  Fail the tool run because something
                        # terrible is happening.
                        LOGGER.debug(
                            f"Calculate Locations chunk {failed_range} continued to error after {num_retries} retries.")
                        LOGGER.error("Failed to get Calculate Locations result from parallel processing.")
                        errs = traceback.format_exc().splitlines()
                        for err in errs:
                            LOGGER.error(err)
                        raise

                # If we got this far, the job completed successfully and we retrieved results.
                completed_jobs += 1
                LOGGER.info(
                    f"Finished Calculate Locations chunk {completed_jobs} of {self.total_jobs}.")

                # Parse the results dictionary and store components for post-processing.
                self.temp_out_fcs.append(result["outputFC"])


if __name__ == "__main__":
    # This script should always be launched via subprocess as if it were being called from the command line.
    launch_parallel_calc_locs()
