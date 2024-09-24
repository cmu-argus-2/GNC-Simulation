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
        job_directory,
        plot_directory,
        PERCENTAGE_OF_DATA_TO_PLOT,
        close_after_saving=True,
    ):
        self.job_dir = job_directory
        self.plot_dir = plot_directory
        self.PERCENTAGE_OF_DATA_TO_PLOT = PERCENTAGE_OF_DATA_TO_PLOT
        self.close_after_saving = close_after_saving

    def position_plots(self):
        true_XYZt_plot = itm.figure()

        data_dict = parse_bin_file(os.path.join(self.job_dir, "position_ECI_true.bin"))
        itm.figure(true_XYZt_plot)
        triPlot(
            data_dict["date [days]"],
            [data_dict["x [m]"], data_dict["y [m]"], data_dict["z [m]"]],
        )

        annotateTriPlot(title="True Position (ECI) [m]")
        save_figure(
            itm.gcf(), self.plot_dir, "position_ECI_true.png", self.close_after_saving
        )
