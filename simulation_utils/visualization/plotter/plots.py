from plot_helper import (
    triPlot,
    annotateTriPlot,
    save_figure,
)
from isolated_trace import itm
from parse_bin_file import parse_bin_file
import os


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
        data_dict = parse_bin_file(os.path.join(self.job_dir, "state_true.bin"))  # noqa: F821

        # ==========================================================================
        itm.figure()
        triPlot(
            data_dict["date [days]"],
            [data_dict["x [m]"], data_dict["y [m]"], data_dict["z [m]"]],
        )
        annotateTriPlot(title="True Position (ECI) [m]")
        save_figure(
            itm.gcf(), self.plot_dir, "position_ECI_true.png", self.close_after_saving
        )
        # ==========================================================================
        itm.figure()
        triPlot(
            data_dict["date [days]"],
            [data_dict["x [m/s]"], data_dict["y [m/s]"], data_dict["z [m/s]"]],
        )
        annotateTriPlot(title="True Velocity (ECI) [m/s]")
        save_figure(
            itm.gcf(), self.plot_dir, "velocity_ECI_true.png", self.close_after_saving
        )
        # ==========================================================================
        # TODO plot attitude
        # ==========================================================================
        itm.figure()
        triPlot(
            data_dict["date [days]"],
            [data_dict["x [rad/s]"], data_dict["y [rad/s]"], data_dict["z [rad/s]"]],
        )
        annotateTriPlot(title="True body angular rate [rad/s]")
        save_figure(
            itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving
        )
