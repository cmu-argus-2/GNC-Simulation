import matplotlib.pyplot as plt
import yaml
from multiprocessing import Pool
import time
from plot_helper import (
    triPlot,
    annotateTriPlot,
    plot_hist,
    save_figure,
    draw_horizontal_line,
)
from isolated_trace import itm
import numpy as np
from parse_bin_file import parse_bin_file, parse_bin_file_wrapper
import os
from plot_state import plot_state, plot_state_error


class MontecarloPlots:
    def __init__(
        self,
        trials,
        trials_directory,
        plot_directory,
        PERCENTAGE_OF_DATA_TO_PLOT,
        close_after_saving=True,
    ):
        self.trials = trials
        self.trials_dir = trials_directory
        self.plot_dir = plot_directory
        self.PERCENTAGE_OF_DATA_TO_PLOT = PERCENTAGE_OF_DATA_TO_PLOT
        self.close_after_saving = close_after_saving
        self.NUM_TRIALS = len(self.trials)

    def position_plot(self):
        position_figure = itm.figure()
        itm.subplot(3, 1, 1)
        itm.title(f"True position [m]")

        filenames = []
        for trial_number in self.trials:
            filenames.append(
                os.path.join(self.trials_dir, f"trial{trial_number}/position_truth.bin")
            )

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        START = time.time()
        for i, trial_number in enumerate(self.trials):
            trial_number = i + 1
            triPlot(
                data_dicts[i]["time [s]"],
                [
                    data_dicts[i]["x [m]"],
                    data_dicts[i]["y [m]"],
                    data_dicts[i]["z [m]"],
                ],
                seriesLabel=f"_{trial_number}",
            )
        save_figure(
            position_figure,
            self.plot_dir,
            "position_truth.png",
            self.close_after_saving,
        )

        END = time.time()
        print(f"Elapsed time to plot: {END-START:.2f} s")

    def velocity_plot(self):
        velocity_figure = itm.figure()
        itm.subplot(3, 1, 1)
        itm.title(f"True velocity [m/s]")

        filenames = []
        for trial_number in self.trials:
            filenames.append(
                os.path.join(self.trials_dir, f"trial{trial_number}/velocity_truth.bin")
            )

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        START = time.time()
        for i, trial_number in enumerate(self.trials):
            trial_number = i + 1
            triPlot(
                data_dicts[i]["time [s]"],
                [
                    data_dicts[i]["x [m/s]"],
                    data_dicts[i]["y [m/s]"],
                    data_dicts[i]["z [m/s]"],
                ],
                seriesLabel=f"_{trial_number}",
            )
        save_figure(
            velocity_figure,
            self.plot_dir,
            "velocity_truth.png",
            self.close_after_saving,
        )

        END = time.time()
        print(f"Elapsed time to plot: {END-START:.2f} s")

    def omega_plot(self):
        omega_figure = itm.figure()
        itm.subplot(3, 1, 1)
        itm.title(f"True omega [deg/s]")

        filenames = []
        for trial_number in self.trials:
            filenames.append(
                os.path.join(self.trials_dir, f"trial{trial_number}/omega_truth.bin")
            )

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        START = time.time()
        for i, trial_number in enumerate(self.trials):
            trial_number = i + 1
            triPlot(
                data_dicts[i]["time [s]"],
                [
                    data_dicts[i]["x [deg/s]"],
                    data_dicts[i]["y [deg/s]"],
                    data_dicts[i]["z [deg/s]"],
                ],
                seriesLabel=f"_{trial_number}",
            )
        save_figure(
            omega_figure,
            self.plot_dir,
            "omega_truth.png",
            self.close_after_saving,
        )

        END = time.time()
        print(f"Elapsed time to plot: {END-START:.2f} s")

    def attitude_plot(self):
        attitude_figure = itm.figure()
        itm.subplot(3, 1, 1)
        itm.title(f"True attitude [deg]")

        filenames = []
        for trial_number in self.trials:
            filenames.append(
                os.path.join(self.trials_dir, f"trial{trial_number}/attitude_truth.bin")
            )

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        START = time.time()
        for i, trial_number in enumerate(self.trials):
            trial_number = i + 1
            triPlot(
                data_dicts[i]["time [s]"],
                [
                    data_dicts[i]["roll [deg]"],
                    data_dicts[i]["pitch [deg]"],
                    data_dicts[i]["yaw [deg]"],
                ],
                seriesLabel=f"_{trial_number}",
            )
        save_figure(
            attitude_figure,
            self.plot_dir,
            "attitude_truth.png",
            self.close_after_saving,
        )

        END = time.time()
        print(f"Elapsed time to plot: {END-START:.2f} s")
