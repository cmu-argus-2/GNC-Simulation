import time
import numpy as np
import struct
import os
import math

from argusim.visualization.plots import *
import argparse

PERCENTAGE_TO_PLOT = 1

def plot_all(result_folder_path: str):
    data = parse_bin_file(os.path.join(result_folder_path, 'state_true.bin'))

    ground_track(data, result_folder_path)
    pos_plot(data, result_folder_path)
    attitude_plot(data, result_folder_path)
    omega_plot(data, result_folder_path)
    bias_plot(data, result_folder_path)
    input_plot(data, result_folder_path)
    sun_point_plot(result_folder_path, data, result_folder_path)
    true_sun_plot(data, result_folder_path)
    true_mag_plot(data, result_folder_path)
    battery_diagnostics_plot(data, result_folder_path)

    

def parse_bin_file(filepath):
    assert 0 < PERCENTAGE_TO_PLOT and PERCENTAGE_TO_PLOT <= 100

    file_size = os.path.getsize(filepath)
    # print(f"{filepath} size: {file_size}")
    with open(filepath, "rb") as file:
        # read in the header
        header = file.readline().decode("utf-8")
        header = header.strip()  # remove leading and trailling whitespace
        header = header.strip(",")  # remove possible trailing comma
        column_labels = header.split(",")
        num_columns = len(column_labels)

        # Compute the number of rows
        byte_position_of_start_of_data_section = file.tell()
        number_of_data_bytes = file_size - byte_position_of_start_of_data_section
        bytes_per_row = 8 * num_columns
        num_rows = int(number_of_data_bytes / bytes_per_row)

        percentage_of_data_to_skip = 100 - PERCENTAGE_TO_PLOT
        rows_to_skip_for_every_row_kept = percentage_of_data_to_skip // PERCENTAGE_TO_PLOT

        start = time.time()
        if rows_to_skip_for_every_row_kept == 0:  # Don't skip any rows
            A = np.zeros((num_rows, num_columns))

            # Read the floating-point data
            for i in range(num_rows):
                data_bytes = file.read(num_columns * 8)  # Assuming double size is 8 bytes
                A[i] = struct.unpack(f"{num_columns}d", data_bytes)
        else:
            num_rows_to_keep = math.ceil(num_rows * PERCENTAGE_TO_PLOT / 100.0)
            A = np.zeros((num_rows_to_keep, num_columns))

            bytes_to_skip_between_kept_rows = rows_to_skip_for_every_row_kept * bytes_per_row

            # Read the floating-point data
            for i in range(num_rows_to_keep):
                data_bytes = file.read(num_columns * 8)  # Assuming double size is 8 bytes
                A[i] = struct.unpack(f"{num_columns}d", data_bytes)
                file.seek(bytes_to_skip_between_kept_rows, 1)
        end = time.time()

        if PERCENTAGE_TO_PLOT == 100:
            print(f"Took {(end - start):.2g} seconds to parse {filepath} ({(number_of_data_bytes/(1024.0**2)):.2g} MB)")
        else:
            print(
                f"Took {(end - start):.2g} seconds to parse every {1+rows_to_skip_for_every_row_kept} rows from {filepath} (kept {((num_rows_to_keep*bytes_per_row)/(1024.0**2)):.2g}/{(number_of_data_bytes/(1024.0**2)):.2g} MB)"
            )

        data_dictionary = {}
        for col, label in zip(A.T, column_labels):
            data_dictionary[label] = col

        return data_dictionary
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="This program generates plots from data produced by a montecarlo job",
        epilog="=" * 80,
    )
    parser.add_argument("job_directory", metavar="job_directory")

    args = parser.parse_args()

    plot_all(args.job_directory)