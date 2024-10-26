from multiprocessing import Pool
import time
from plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from isolated_trace import itm
import numpy as np
from parse_bin_file import parse_bin_file_wrapper
from mpl_toolkits.basemap import Basemap
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
        m = Basemap()  # cylindrical projection by default
        m.bluemarble(scale=0.1, alpha=0.4)
        m.drawcoastlines(linewidth=0.5)
        m.fillcontinents(color="0.8")
        m.drawparallels(np.arange(-90, 90, 15), linewidth=0.2, labels=[1, 1, 0, 0])
        m.drawmeridians(np.arange(-180, 180, 30), linewidth=0.2, labels=[0, 0, 0, 1])

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

            # TODO convert from ECI to ECEF
            lon = np.rad2deg(np.arctan2(y_km, x_km))
            lat = np.rad2deg(np.arctan(z_km / np.hypot(x_km, y_km)))

            # https://matplotlib.org/basemap/stable/users/examples.html
            itm.figure(ground_track)
            m.scatter(lon, lat, s=0.1, marker=".", label=f"_{trial_number}", latlon=True)
            m.scatter(lon[0], lat[0], marker="*", color="green", label="Start")
            m.scatter(lon[-1], lat[-1], marker="*", color="red", label="End")
        itm.figure(XYZt)
        annotateMultiPlot(title="True Position (ECI) [km]", ylabels=["x", "y", "z"])
        save_figure(XYZt, self.plot_dir, "position_ECI_true.png", self.close_after_saving)

        itm.figure(ground_track)
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
