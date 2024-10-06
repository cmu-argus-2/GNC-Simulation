from multiprocessing import Pool
import time
from plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from isolated_trace import itm
import numpy as np
from parse_bin_file import parse_bin_file, parse_bin_file_wrapper
import os


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

    def true_state_plots(self):
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/state_true.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        # ==========================================================================
        XYZt = itm.figure()
        ground_track = itm.figure()
        for i, trial_number in enumerate(self.trials):
            itm.figure(XYZt)

            x_km = data_dicts[i]["x ECI [m]"] / 1000
            y_km = data_dicts[i]["y ECI [m]"] / 1000
            z_km = data_dicts[i]["z ECI [m]"] / 1000
            multiPlot(
                data_dicts[i]["time [s]"],
                [x_km, y_km, z_km],
                seriesLabel=f"_{trial_number}",
            )

            itm.figure(ground_track)
            lon = np.rad2deg(np.arctan2(y_km, x_km))
            lat = np.rad2deg(np.arctan(z_km / np.hypot(x_km, y_km)))
            itm.scatter(lon, lat, s=0.1, label=f"_{trial_number}")
            itm.scatter(lon[0], lat[0], marker="*", color="green", label="Start")
            itm.scatter(lon[-1], lat[-1], marker="*", color="red", label="End")

        itm.figure(XYZt)
        annotateMultiPlot(title="True Position (ECI) [km]", ylabels=["x", "y", "z"])
        save_figure(XYZt, self.plot_dir, "position_ECI_true.png", self.close_after_saving)

        itm.figure(ground_track)
        itm.xlabel("Longitude [deg]")
        itm.ylabel("Latitude [deg]")
        itm.xlim([-185, 185])
        itm.ylim([-95, 95])
        itm.gca().set_aspect("equal")
        itm.title("Ground Track [Green: Start    Red: End]")
        save_figure(ground_track, self.plot_dir, "ground_track.png", self.close_after_saving)
        # ==========================================================================
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["time [s]"],
                np.array([data_dicts[i]["x ECI [m/s]"], data_dicts[i]["y ECI [m/s]"], data_dicts[i]["z ECI [m/s]"]])
                / 1000,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True Velocity (ECI) [km/s]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "velocity_ECI_true.png", self.close_after_saving)
        # ==========================================================================
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["time [s]"],
                [data_dicts[i]["w"], data_dicts[i]["x"], data_dicts[i]["y"], data_dicts[i]["z"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True attitude [-]", ylabels=["w", "x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "attitude_true.png", self.close_after_saving)
        # ==========================================================================
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["time [s]"],
                np.rad2deg([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]]),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True body angular rate [deg/s]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving)
