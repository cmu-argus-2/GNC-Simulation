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
from matplotlib import pyplot as plt

# TODO : Setup Python setuptools
import sys

sys.path.append("../../")
from build.world.pyphysics import ECI2GEOD


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

    def true_state_plots(self):
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/states.bin"))

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
            x_km = data_dicts[i]["r_x ECI [m]"] / 1000
            y_km = data_dicts[i]["r_y ECI [m]"] / 1000
            z_km = data_dicts[i]["r_z ECI [m]"] / 1000
            time_vec = data_dicts[i]["Time [s]"]
            multiPlot(
                time_vec,
                [x_km, y_km, z_km],
                seriesLabel=f"_{trial_number}",
            )

            # TODO convert from ECI to ECEF
            lon = np.zeros_like(x_km)
            lat = np.zeros_like(x_km)
            for k in range(len(x_km)):
                lon[k], lat[k], _ = ECI2GEOD([x_km[k] * 1000, y_km[k] * 1000, z_km[k] * 1000], time_vec[k])

            # https://matplotlib.org/basemap/stable/users/examples.html
            itm.figure(ground_track)
            m.scatter(lon, lat, s=0.1, marker=".", label=f"_{trial_number}", latlon=True)
            m.scatter(lon[0], lat[0], marker="*", color="green", label="Start")
            m.scatter(lon[-1], lat[-1], marker="*", color="red", label="End")
        itm.figure(XYZt)
        annotateMultiPlot(title="True Position (ECI) [km]", ylabels=["$r_x$", "$r_y$", "$r_z$"])
        save_figure(XYZt, self.plot_dir, "position_ECI_true.png", self.close_after_saving)

        itm.figure(ground_track)
        itm.gca().set_aspect("equal")
        itm.title("Ground Track [Green: Start    Red: End]")
        save_figure(ground_track, self.plot_dir, "ground_track.png", self.close_after_saving)
        # ==========================================================================
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.array(
                    [data_dicts[i]["v_x ECI [m/s]"], data_dicts[i]["v_y ECI [m/s]"], data_dicts[i]["v_z ECI [m/s]"]]
                )
                / 1000,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True Velocity (ECI) [km/s]", ylabels=["$v_x$", "$v_y$", "$v_z$"])
        save_figure(itm.gcf(), self.plot_dir, "velocity_ECI_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
        save_figure(itm.gcf(), self.plot_dir, "attitude_true.png", self.close_after_saving)

        # Estimated
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [
                    data_dicts[i]["q_hat_w"],
                    data_dicts[i]["q_hat_x"],
                    data_dicts[i]["q_hat_y"],
                    data_dicts[i]["q_hat_z"],
                ],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Estimated attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
        save_figure(itm.gcf(), self.plot_dir, "attitude_estimate.png", self.close_after_saving)
        # ==========================================================================
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(
                    [
                        data_dicts[i]["omega_x [rad/s]"],
                        data_dicts[i]["omega_y [rad/s]"],
                        data_dicts[i]["omega_z [rad/s]"],
                    ]
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True body angular rate [deg/s]", ylabels=["$\Omega_x$", "$\Omega_y$", "$\Omega_z$"])
        save_figure(itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving)

    def EKF_plots(self):
        # error plots
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/attitude_ekf_error.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        # ==========================================================================
        self.attitude_error_figure = itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(np.array([data_dicts[i]["x [rad]"], data_dicts[i]["y [rad]"], data_dicts[i]["z [rad]"]])),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Attitude error [deg]", ylabels=["$x$", "$y$", "$z$"])
        # ==========================================================================
        self.gyro_bias_error_figure = itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(
                    np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Gyro Bias error [deg/s]", ylabels=["$x$", "$y$", "$z$"])

        self._plot_state_estimate_covariance()
        # ========================= True gyro bias plots =========================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/gyro_bias_true.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(
                    np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True Gyro Bias [deg/s]", ylabels=["$x$", "$y$", "$z$"])
        save_figure(itm.gcf(), self.plot_dir, "gyro_bias_true.png", self.close_after_saving)

        # ======================= Estimated gyro bias plots =======================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/gyro_bias_estimated.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(
                    np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Estimated Gyro Bias [deg/s]", ylabels=["$x$", "$y$", "$z$"])
        save_figure(itm.gcf(), self.plot_dir, "gyro_bias_estimated.png", self.close_after_saving)

    def sensor_measurement_plots(self):
        # ======================= Gyro measurement plots =======================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/gyro_measurement.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(
                    np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Gyro measurement [deg/s]", ylabels=["$\Omega_x$", "$\Omega_y$", "$\Omega_z$"])
        save_figure(itm.gcf(), self.plot_dir, "gyro_measurement.png", self.close_after_saving)
        # ====================== Sun Sensor measurement plots ======================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/sun_sensor_measurement.bin"))

        START = time.time()
        args = [(filename, 100) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.array([data_dicts[i]["x [-]"], data_dicts[i]["y [-]"], data_dicts[i]["z [-]"]]),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Measured Sun Ray in body frame", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "sun_sensor_body_measurement.png", self.close_after_saving)

    def _plot_state_estimate_covariance(self):
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/state_covariance.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

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
        num_datapoints = [len(data_dicts[i]["Time [s]"]) for i in range(len(self.trials))]
        N_max = max(num_datapoints)

        for i, trial_number in enumerate(self.trials):
            data_dict = data_dicts[i]

        t = None
        # fmt: off
        for i, trial_number in enumerate(self.trials):
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
        mean_sigma_attitude_x = summed_sigma_attitude_x / self.NUM_TRIALS
        mean_sigma_attitude_y = summed_sigma_attitude_y / self.NUM_TRIALS
        mean_sigma_attitude_z = summed_sigma_attitude_z / self.NUM_TRIALS
        mean_sigma_gyro_bias_x = summed_sigma_gyro_bias_x / self.NUM_TRIALS
        mean_sigma_gyro_bias_y = summed_sigma_gyro_bias_y / self.NUM_TRIALS
        mean_sigma_gyro_bias_z = summed_sigma_gyro_bias_z / self.NUM_TRIALS

        itm.figure(self.attitude_error_figure)
        multiPlot(t, [3 * min_sigma_attitude_x, 3 * min_sigma_attitude_y, 3 * min_sigma_attitude_z], linestyle="-.", linewidth=1, color='g', seriesLabel=r"3$\sigma$ (min)")
        multiPlot(t, [3 * mean_sigma_attitude_x, 3 * mean_sigma_attitude_y, 3 * mean_sigma_attitude_z], linestyle="-.", linewidth=1, color='k', seriesLabel=r"3$\sigma$ (mean)")
        multiPlot(t, [3 * max_sigma_attitude_x, 3 * max_sigma_attitude_y, 3 * max_sigma_attitude_z], linestyle="-.", linewidth=1, color='r', seriesLabel=r"3$\sigma$ (max)")
        multiPlot(t, [-3 * min_sigma_attitude_x, -3 * min_sigma_attitude_y, -3 * min_sigma_attitude_z], linestyle="-.", linewidth=1, color='g')
        multiPlot(t, [-3 * mean_sigma_attitude_x, -3 * mean_sigma_attitude_y, -3 * mean_sigma_attitude_z], linestyle="-.", linewidth=1, color='k')
        multiPlot(t, [-3 * max_sigma_attitude_x, -3 * max_sigma_attitude_y, -3 * max_sigma_attitude_z], linestyle="-.", linewidth=1, color='r')

        itm.figure(self.gyro_bias_error_figure)
        multiPlot(t, [3 * min_sigma_gyro_bias_x, 3 * min_sigma_gyro_bias_y, 3 * min_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='g', seriesLabel=r"3$\sigma$ (min)")
        multiPlot(t, [3 * mean_sigma_gyro_bias_x, 3 * mean_sigma_gyro_bias_y, 3 * mean_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='k', seriesLabel=r"3$\sigma$ (mean)")
        multiPlot(t, [3 * max_sigma_gyro_bias_x, 3 * max_sigma_gyro_bias_y, 3 * max_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='r', seriesLabel=r"3$\sigma$ (max)")
        multiPlot(t, [-3 * min_sigma_gyro_bias_x, -3 * min_sigma_gyro_bias_y, -3 * min_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='g')
        multiPlot(t, [-3 * mean_sigma_gyro_bias_x, -3 * mean_sigma_gyro_bias_y, -3 * mean_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='k')
        multiPlot(t, [-3 * max_sigma_gyro_bias_x, -3 * max_sigma_gyro_bias_y, -3 * max_sigma_gyro_bias_z], linestyle="-.", linewidth=1, color='r')

        # fmt: on

        save_figure(self.attitude_error_figure, self.plot_dir, "attitude_estimate_error.png", self.close_after_saving)
        save_figure(self.gyro_bias_error_figure, self.plot_dir, "gyro_bias_estimate_error.png", self.close_after_saving)
