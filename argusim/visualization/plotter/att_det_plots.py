
import numpy as np
import matplotlib.pyplot as plt
from argusim.visualization.plotter.isolated_trace import itm
import time
import yaml
import os
from argusim.world.math.quaternions import quatrotation
from argusim.visualization.plotter.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from argusim.actuators import Magnetorquer


def plot_state_est_cov(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving     = pyparams["close_after_saving"]
    gyro_bias_error_figure = pyparams["gyro_bias_error_figure"]
    attitude_error_figure  = pyparams["attitude_error_figure"]
    NUM_TRIALS             = pyparams["NUM_TRIALS"]

    # point-wise minimum std-dev across all self.trials at each point in time
    min_sigma_attitude_x = None
    min_sigma_attitude_y = None
    min_sigma_attitude_z = None
    min_sigma_gyro_bias_x = None
    min_sigma_gyro_bias_y = None
    min_sigma_gyro_bias_z = None

    summed_sigma_attitude_x = None
    summed_sigma_attitude_y = None
    summed_sigma_attitude_z = None
    summed_sigma_gyro_bias_x = None
    summed_sigma_gyro_bias_y = None
    summed_sigma_gyro_bias_z = None

    # point-wise maximum std-dev across all self.trials at each point in time
    max_sigma_attitude_x = None
    max_sigma_attitude_y = None
    max_sigma_attitude_z = None
    max_sigma_gyro_bias_x = None
    max_sigma_gyro_bias_y = None
    max_sigma_gyro_bias_z = None

    # self.trials might have slightly different lengths; identify length of the
    # longest trial and pad the other time series to match that length
    num_datapoints = [len(data_dicts[i]["Time [s]"]) for i in range(len(data_dicts))]
    N_max = max(num_datapoints)

    for i, (trial_number, _) in enumerate(filepaths):
        data_dict = data_dicts[i]

    t = None
    # fmt: off
    for i, (trial_number, _) in enumerate(filepaths):
        data_dict = data_dicts[i]
        N = num_datapoints[i]
        if t is None and N == N_max:
            t = data_dict["Time [s]"]

        padding_length = N_max - N
        padding = np.empty((padding_length))
        padding[:] = np.nan

        sigma_attitude_x = np.concatenate((np.rad2deg(data_dict["attitude error x [rad]"]), padding))
        sigma_attitude_y = np.concatenate((np.rad2deg(data_dict["attitude error y [rad]"]), padding))
        sigma_attitude_z = np.concatenate((np.rad2deg(data_dict["attitude error z [rad]"]), padding))
        sigma_gyro_bias_x = np.concatenate((np.rad2deg(data_dict["gyro bias error x [rad/s]"]), padding))
        sigma_gyro_bias_y = np.concatenate((np.rad2deg(data_dict["gyro bias error y [rad/s]"]), padding))
        sigma_gyro_bias_z = np.concatenate((np.rad2deg(data_dict["gyro bias error z [rad/s]"]), padding))
        # "RuntimeWarning: All-NaN axis encountered" is expected
        min_sigma_attitude_x = sigma_attitude_x if min_sigma_attitude_x is None else np.nanmin((min_sigma_attitude_x, sigma_attitude_x), axis=0)
        min_sigma_attitude_y = sigma_attitude_y if min_sigma_attitude_y is None else np.nanmin((min_sigma_attitude_y, sigma_attitude_y), axis=0)
        min_sigma_attitude_z = sigma_attitude_z if min_sigma_attitude_z is None else np.nanmin((min_sigma_attitude_z, sigma_attitude_z), axis=0)
        min_sigma_gyro_bias_x = sigma_gyro_bias_x if min_sigma_gyro_bias_x is None else np.nanmin((min_sigma_gyro_bias_x, sigma_gyro_bias_x), axis=0)
        min_sigma_gyro_bias_y = sigma_gyro_bias_y if min_sigma_gyro_bias_y is None else np.nanmin((min_sigma_gyro_bias_y, sigma_gyro_bias_y), axis=0)
        min_sigma_gyro_bias_z = sigma_gyro_bias_z if min_sigma_gyro_bias_z is None else np.nanmin((min_sigma_gyro_bias_z, sigma_gyro_bias_z), axis=0)

        summed_sigma_attitude_x = sigma_attitude_x if summed_sigma_attitude_x is None else np.nansum((summed_sigma_attitude_x, sigma_attitude_x), axis=0)
        summed_sigma_attitude_y = sigma_attitude_y if summed_sigma_attitude_y is None else np.nansum((summed_sigma_attitude_y, sigma_attitude_y), axis=0)
        summed_sigma_attitude_z = sigma_attitude_z if summed_sigma_attitude_z is None else np.nansum((summed_sigma_attitude_z, sigma_attitude_z), axis=0)
        summed_sigma_gyro_bias_x = sigma_gyro_bias_x if summed_sigma_gyro_bias_x is None else np.nansum((summed_sigma_gyro_bias_x, sigma_gyro_bias_x), axis=0)
        summed_sigma_gyro_bias_y = sigma_gyro_bias_y if summed_sigma_gyro_bias_y is None else np.nansum((summed_sigma_gyro_bias_y, sigma_gyro_bias_y), axis=0)
        summed_sigma_gyro_bias_z = sigma_gyro_bias_z if summed_sigma_gyro_bias_z is None else np.nansum((summed_sigma_gyro_bias_z, sigma_gyro_bias_z), axis=0)

        max_sigma_attitude_x = sigma_attitude_x if max_sigma_attitude_x is None else np.nanmax((max_sigma_attitude_x, sigma_attitude_x), axis=0)
        max_sigma_attitude_y = sigma_attitude_y if max_sigma_attitude_y is None else np.nanmax((max_sigma_attitude_y, sigma_attitude_y), axis=0)
        max_sigma_attitude_z = sigma_attitude_z if max_sigma_attitude_z is None else np.nanmax((max_sigma_attitude_z, sigma_attitude_z), axis=0)
        max_sigma_gyro_bias_x = sigma_gyro_bias_x if max_sigma_gyro_bias_x is None else np.nanmax((max_sigma_gyro_bias_x, sigma_gyro_bias_x), axis=0)
        max_sigma_gyro_bias_y = sigma_gyro_bias_y if max_sigma_gyro_bias_y is None else np.nanmax((max_sigma_gyro_bias_y, sigma_gyro_bias_y), axis=0)
        max_sigma_gyro_bias_z = sigma_gyro_bias_z if max_sigma_gyro_bias_z is None else np.nanmax((max_sigma_gyro_bias_z, sigma_gyro_bias_z), axis=0)

    # point-wise mean std-dev across all self.trials at each poitn in time
    mean_sigma_attitude_x = summed_sigma_attitude_x / NUM_TRIALS
    mean_sigma_attitude_y = summed_sigma_attitude_y / NUM_TRIALS
    mean_sigma_attitude_z = summed_sigma_attitude_z / NUM_TRIALS
    mean_sigma_gyro_bias_x = summed_sigma_gyro_bias_x / NUM_TRIALS
    mean_sigma_gyro_bias_y = summed_sigma_gyro_bias_y / NUM_TRIALS
    mean_sigma_gyro_bias_z = summed_sigma_gyro_bias_z / NUM_TRIALS

    itm.figure(attitude_error_figure)
    multiPlot(t, [3 * min_sigma_attitude_x, 3 * min_sigma_attitude_y, 3 * min_sigma_attitude_z], linestyle="-.", linewidth=1, color='g', seriesLabel=r"3$\sigma$ (min)")
    multiPlot(t, [3 * mean_sigma_attitude_x, 3 * mean_sigma_attitude_y, 3 * mean_sigma_attitude_z], linestyle="-.", linewidth=1, color='k', seriesLabel=r"3$\sigma$ (mean)")
    multiPlot(t, [3 * max_sigma_attitude_x, 3 * max_sigma_attitude_y, 3 * max_sigma_attitude_z], linestyle="-.", linewidth=1, color='r', seriesLabel=r"3$\sigma$ (max)")
    multiPlot(t, [-3 * min_sigma_attitude_x, -3 * min_sigma_attitude_y, -3 * min_sigma_attitude_z], linestyle="-.", linewidth=1, color='g')
    multiPlot(t, [-3 * mean_sigma_attitude_x, -3 * mean_sigma_attitude_y, -3 * mean_sigma_attitude_z], linestyle="-.", linewidth=1, color='k')
    multiPlot(t, [-3 * max_sigma_attitude_x, -3 * max_sigma_attitude_y, -3 * max_sigma_attitude_z], linestyle="-.", linewidth=1, color='r')
    for i in range(3):
        itm.subplot(3, 1, i + 1)
        itm.legend(loc = 'upper right')

    itm.figure(gyro_bias_error_figure)
    multiPlot(t, [3 * min_sigma_gyro_bias_x, 3 * min_sigma_gyro_bias_y, 3 * min_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='g', seriesLabel=r"3$\sigma$ (min)")
    multiPlot(t, [3 * mean_sigma_gyro_bias_x, 3 * mean_sigma_gyro_bias_y, 3 * mean_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='k', seriesLabel=r"3$\sigma$ (mean)")
    multiPlot(t, [3 * max_sigma_gyro_bias_x, 3 * max_sigma_gyro_bias_y, 3 * max_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='r', seriesLabel=r"3$\sigma$ (max)")
    multiPlot(t, [-3 * min_sigma_gyro_bias_x, -3 * min_sigma_gyro_bias_y, -3 * min_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='g')
    multiPlot(t, [-3 * mean_sigma_gyro_bias_x, -3 * mean_sigma_gyro_bias_y, -3 * mean_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='k')
    multiPlot(t, [-3 * max_sigma_gyro_bias_x, -3 * max_sigma_gyro_bias_y, -3 * max_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='r')
    for i in range(3):
        itm.subplot(3, 1, i + 1)
        itm.legend(loc = 'upper right')

    # fmt: on

    save_figure(attitude_error_figure, plot_dir, "attitude_estimate_error.png", close_after_saving)
    save_figure(gyro_bias_error_figure, plot_dir, "gyro_bias_estimate_error.png", close_after_saving)


def EKF_err_plots(pyparams, data_dicts):
    if not pyparams["PlotFlags"]["MEKF_plots"]:
        return
    trials = pyparams["trials"] 
    close_after_saving = pyparams["close_after_saving"] 
    plot_dir = pyparams["plot_dir"] 
    # ==========================================================================
    # Attitude error plots
    pyparams["attitude_error_figure"] = itm.figure()
    final_attitude_error_norms = []
    for i, trial_number in enumerate(trials):
        series = np.rad2deg(
            np.array([data_dicts[i]["x [rad]"], data_dicts[i]["y [rad]"], data_dicts[i]["z [rad]"]])
        )
        multiPlot(data_dicts[i]["Time [s]"], series, seriesLabel=f"_{trial_number}")
        final_attitude_error_norms.append(np.linalg.norm(series[:, -1]))
    annotateMultiPlot(title="Attitude error [deg]", ylabels=["$x$", "$y$", "$z$"])

    itm.figure()
    itm.hist(
        final_attitude_error_norms,
        bins=np.arange(min(final_attitude_error_norms), max(final_attitude_error_norms) + 1, 1),
        density=True,
        # cumulative=True,
        edgecolor="black",
    )
    percentile_95 = np.percentile(final_attitude_error_norms, 95)
    itm.axvline(x=percentile_95, color="red", linestyle="--", label=f"95th Percentile: {percentile_95:.2f} [deg]")
    itm.title("RSS Final Attitude Estimate Error - PDF")
    itm.xlabel("[deg]")
    # itm.ylabel("%-tile")
    itm.legend(fontsize=20)
    save_figure(itm.gcf(), plot_dir, "attitude_estimate_error_hist.png", close_after_saving)
    # ==========================================================================
    # Gyro Bias error plots
    pyparams["gyro_bias_error_figure"] = itm.figure()
    final_gyro_bias_error_norms = []
    for i, trial_number in enumerate(trials):
        series = np.rad2deg(
            np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
        )
        multiPlot(data_dicts[i]["Time [s]"], series, seriesLabel=f"_{trial_number}")
        final_gyro_bias_error_norms.append(np.linalg.norm(series[:, -1]))
    annotateMultiPlot(title="Gyro Bias error [deg/s]", ylabels=["$x$", "$y$", "$z$"])

    itm.figure()
    itm.hist(
        final_gyro_bias_error_norms,
        bins=np.arange(min(final_gyro_bias_error_norms), max(final_gyro_bias_error_norms) + 0.1, 0.1),
        density=True,
        # cumulative=True,
        edgecolor="black",
    )
    percentile_95 = np.percentile(final_gyro_bias_error_norms, 95)
    itm.axvline(x=percentile_95, color="red", linestyle="--", label=f"95th Percentile: {percentile_95:.2f} [deg/s]")
    itm.title("RSS Final Gyro Bias Estimate Error - PDF")
    itm.xlabel("[deg/s]")
    # itm.ylabel("%-tile")
    itm.legend(fontsize=20)
    save_figure(itm.gcf(), plot_dir, "gyro_bias_estimate_error_hist.png", close_after_saving)

    return pyparams

def EKF_st_plots(pyparams, data_dicts, filepaths):
    if not pyparams["PlotFlags"]["MEKF_plots"]:
        return
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ==========================================================================
    # Estimated Attitude
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            [data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]],
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Estimated attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
    save_figure(itm.gcf(), plot_dir, "attitude_estimated.png", close_after_saving)
    # ==========================================================================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.rad2deg(
                np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
            ),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Estimated Gyro Bias [deg/s]", ylabels=["$x$", "$y$", "$z$"])
    save_figure(itm.gcf(), plot_dir, "gyro_bias_estimated.png", close_after_saving)