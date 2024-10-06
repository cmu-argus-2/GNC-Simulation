import struct
import time
import numpy as np
import os
import math


# this wrapper is useful for multiprocessing several binary files concurrently
def parse_bin_file_wrapper(args):
    return parse_bin_file(*args)


def parse_bin_file(filepath, percentage_of_data_to_keep=100):
    assert 0 < percentage_of_data_to_keep and percentage_of_data_to_keep <= 100

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

        percentage_of_data_to_skip = 100 - percentage_of_data_to_keep
        rows_to_skip_for_every_row_kept = percentage_of_data_to_skip // percentage_of_data_to_keep

        start = time.time()
        if rows_to_skip_for_every_row_kept == 0:  # Don't skip any rows
            A = np.zeros((num_rows, num_columns))

            # Read the floating-point data
            for i in range(num_rows):
                data_bytes = file.read(num_columns * 8)  # Assuming double size is 8 bytes
                A[i] = struct.unpack(f"{num_columns}d", data_bytes)
        else:
            num_rows_to_keep = math.ceil(num_rows * percentage_of_data_to_keep / 100.0)
            A = np.zeros((num_rows_to_keep, num_columns))

            bytes_to_skip_between_kept_rows = rows_to_skip_for_every_row_kept * bytes_per_row

            # Read the floating-point data
            for i in range(num_rows_to_keep):
                data_bytes = file.read(num_columns * 8)  # Assuming double size is 8 bytes
                A[i] = struct.unpack(f"{num_columns}d", data_bytes)
                file.seek(bytes_to_skip_between_kept_rows, 1)
        end = time.time()

        if percentage_of_data_to_keep == 100:
            print(f"Took {(end - start):.2g} seconds to parse {filepath} ({(number_of_data_bytes/(1024.0**2)):.2g} MB)")
        else:
            print(
                f"Took {(end - start):.2g} seconds to parse every {1+rows_to_skip_for_every_row_kept} rows from {filepath} (kept {((num_rows_to_keep*bytes_per_row)/(1024.0**2)):.2g}/{(number_of_data_bytes/(1024.0**2)):.2g} MB)"
            )

        data_dictionary = {}
        for col, label in zip(A.T, column_labels):
            data_dictionary[label] = col

        return data_dictionary
