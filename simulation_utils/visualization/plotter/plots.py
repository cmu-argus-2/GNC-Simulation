from plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from isolated_trace import itm
from parse_bin_file import parse_bin_file
import os

import numpy as np


class MontecarloPlots:
    def __init__(
        self,
        job_directory,
        plot_directory,
        PERCENTAGE_OF_DATA_TO_PLOT,
        close_after_saving=True,
    ):
        self.job_dir = job_directory
        self.plot_dir = plot_directory
        self.PERCENTAGE_OF_DATA_TO_PLOT = PERCENTAGE_OF_DATA_TO_PLOT
        self.close_after_saving = close_after_saving

    def true_state_plots(self):
        data_dict = parse_bin_file(os.path.join(self.job_dir, "state_true.bin"))

        # ==========================================================================
        itm.figure()
        multiPlot(
            data_dict["date [days]"],
            np.array([data_dict["x [m]"], data_dict["y [m]"], data_dict["z [m]"]])
            / 1000,
        )
        annotateMultiPlot(title="True Position (ECI) [km]", ylabels=["x", "y", "z"])
        save_figure(
            itm.gcf(), self.plot_dir, "position_ECI_true.png", self.close_after_saving
        )
        # ==========================================================================
        itm.figure()
        multiPlot(
            data_dict["date [days]"],
            np.array([data_dict["x [m/s]"], data_dict["y [m/s]"], data_dict["z [m/s]"]])
            / 1000,
        )
        annotateMultiPlot(title="True Velocity (ECI) [km/s]", ylabels=["x", "y", "z"])
        save_figure(
            itm.gcf(), self.plot_dir, "velocity_ECI_true.png", self.close_after_saving
        )
        # ==========================================================================
        # TODO plot attitude
        # ==========================================================================
        itm.figure()
        multiPlot(
            data_dict["date [days]"],
            np.rad2deg(
                [data_dict["x [rad/s]"], data_dict["y [rad/s]"], data_dict["z [rad/s]"]]
            ),
        )
        annotateMultiPlot(
            title="True body angular rate [deg/s]", ylabels=["x", "y", "z"]
        )
        save_figure(
            itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving
        )

    def measured_omega(self):
        data_dict = parse_bin_file(os.path.join(self.job_dir, "omega_meausured.bin"))

        # ==========================================================================
        itm.figure()
        multiPlot(
            data_dict["date [days]"],
            np.rad2deg(
                [data_dict["x [rad/s]"], data_dict["y [rad/s]"], data_dict["z [rad/s]"]]
            ),
        )
        annotateMultiPlot(
            title="Measured body angular rate [deg/sec]", ylabels=["x", "y", "z"]
        )
        save_figure(
            itm.gcf(), self.plot_dir, "body_omega_measured.png", self.close_after_saving
        )

    def star_tracker(self):
        data_dict = parse_bin_file(os.path.join(self.job_dir, "J2000_R_ST_true.bin"))
        itm.figure()
        multiPlot(
            data_dict["date [days]"],
            [data_dict["w"], data_dict["x"], data_dict["y"], data_dict["z"]],
        )
        annotateMultiPlot(title="True J2000_R_ST", ylabels=["w", "x", "y", "z"])
        save_figure(
            itm.gcf(), self.plot_dir, "J2000_R_ST_true.png", self.close_after_saving
        )
        # ==========================================================================
        data_dict = parse_bin_file(
            os.path.join(self.job_dir, "J2000_R_ST_measured.bin")
        )
        itm.figure()
        multiPlot(
            data_dict["date [days]"],
            [data_dict["w"], data_dict["x"], data_dict["y"], data_dict["z"]],
        )
        annotateMultiPlot(title="Measured J2000_R_ST", ylabels=["w", "x", "y", "z"])
        save_figure(
            itm.gcf(), self.plot_dir, "J2000_R_ST_measured.png", self.close_after_saving
        )
