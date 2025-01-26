from multiprocessing import Pool
import time
import numpy as np
from mpl_toolkits.basemap import Basemap
import os

from argusim.visualization.plotter.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
from argusim.visualization.plotter.isolated_trace import itm
from argusim.visualization.plotter.parse_bin_file import parse_bin_file_wrapper
from argusim.visualization.plotter.plot_pointing import pointing_plots
from argusim.visualization.plotter.actuator_plots import actuator_plots
from argusim.visualization.plotter.att_animation import att_animation
from argusim.build.world.pyphysics import ECI2GEOD
from argusim.world.math.quaternions import quatrotation
from argusim.actuators import Magnetorquer
import yaml
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes


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

    def _get_files_across_trials(self, filename):
        filepaths = []
        for trial_number in self.trials:
            filepath = os.path.join(self.trials_dir, f"trial{trial_number}/{filename}")
            if os.path.exists(filepath):
                filepaths.append((trial_number, filepath))
            else:
                print(RED + f"Trial {trial_number} is missing {filename}" + RESET)
        return filepaths

    def true_state_plots(self):
        filepaths = self._get_files_across_trials("state_true.bin")

        START = time.time()
        args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")

        # ==========================================================================
        XYZt = itm.figure()

        ground_track = itm.figure()
        m = Basemap()  # cylindrical projection by default
        m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
        m.drawparallels(np.arange(-90, 90, 15), linewidth=0.2, labels=[1, 1, 0, 0])
        m.drawmeridians(np.arange(-180, 180, 30), linewidth=0.2, labels=[0, 0, 0, 1])

        for i, (trial_number, _) in enumerate(filepaths):
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
            m.scatter(lon, lat, s=0.5, c="y", marker=".", label=f"_{trial_number}", latlon=True)
            m.scatter(lon[0], lat[0], marker="*", color="green", label="Start")
            m.scatter(lon[-1], lat[-1], marker="*", color="red", label="End")

        itm.figure(XYZt)
        annotateMultiPlot(title="True Position (ECI) [km]", ylabels=["r_x", "r_y", "r_z"])
        save_figure(XYZt, self.plot_dir, "position_ECI_true.png", self.close_after_saving)

        itm.figure(ground_track)
        itm.gca().set_aspect("equal")
        itm.title("Ground Track [Green: Start    Red: End]")
        save_figure(ground_track, self.plot_dir, "ground_track.png", self.close_after_saving)
        # ==========================================================================
        # Plot the ECI trajectory in 3D and add a vector of the mean sun direction
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        max_range = 0
        for i, trial_number in enumerate(self.trials):
            x_km = data_dicts[i]["r_x ECI [m]"] / 1000
            y_km = data_dicts[i]["r_y ECI [m]"] / 1000
            z_km = data_dicts[i]["r_z ECI [m]"] / 1000
            ax.plot(x_km, y_km, z_km, label=f'Trial {trial_number}', color='red')

            max_range = max(max_range, np.max(np.abs(x_km)), np.max(np.abs(y_km)), np.max(np.abs(z_km)))

            # Mark points where the orbit crosses the xy plane
            crossings = 0
            for j in range(1, len(z_km)):
                if z_km[j-1] * z_km[j] < 0:  # Check for sign change indicating a crossing
                    # Interpolate to find the crossing point
                    t = -z_km[j-1] / (z_km[j] - z_km[j-1])
                    x_cross = x_km[j-1] + t * (x_km[j] - x_km[j-1])
                    y_cross = y_km[j-1] + t * (y_km[j] - y_km[j-1])
                    ax.scatter(x_cross, y_cross, 0, color='blue', s=50, label='XY Plane Crossing')
                    crossings += 1
                    if crossings >= 2:
                        break

                    # Calculate the angle to the sun vector
                    sun_vector = np.array([data_dicts[i]["rSun_x ECI [m]"][j], data_dicts[i]["rSun_y ECI [m]"][j], data_dicts[i]["rSun_z ECI [m]"][j]])
                    sun_vector /= np.linalg.norm(sun_vector)
                    orbit_vector = np.array([x_cross, y_cross, 0])
                    orbit_vector /= np.linalg.norm(orbit_vector)
                    angle_to_sun = np.rad2deg(np.arccos(np.dot(sun_vector, orbit_vector)))
                    ax.text(x_cross, y_cross, 0, f'{angle_to_sun:.1f}°', color='blue')

        # Calculate the mean sun direction
        mean_sun_x = np.mean([data_dicts[i]["rSun_x ECI [m]"] for i in range(len(self.trials))], axis=1)
        mean_sun_y = np.mean([data_dicts[i]["rSun_y ECI [m]"] for i in range(len(self.trials))], axis=1)
        mean_sun_z = np.mean([data_dicts[i]["rSun_z ECI [m]"] for i in range(len(self.trials))], axis=1)
        mean_sun_direction = np.array([mean_sun_x, mean_sun_y, mean_sun_z]).flatten()
        mean_sun_direction /= np.linalg.norm(mean_sun_direction)

        # Plot the mean sun direction vector
        earth_radius_km = 6371
        start_point = mean_sun_direction * earth_radius_km
        ax.quiver(0, 0, 0, mean_sun_direction[0], mean_sun_direction[1], mean_sun_direction[2], 
              length=earth_radius_km, color='orange', label='Mean Sun Direction')

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        ax.legend()
        ax.text(start_point[0], start_point[1], start_point[2], "Sun Direction", color='orange')

        # Set identical limits for all axes
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        ax.view_init(elev=90, azim=0)
        plt.savefig(os.path.join(self.plot_dir, "eci_trajectory_with_mean_sun_direction.png"))
        plt.close(fig)
        # ==========================================================================
        # Truth Velocity (ECI)
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
                np.array([data_dicts[i]["v_x ECI [m/s]"], data_dicts[i]["v_y ECI [m/s]"], data_dicts[i]["v_z ECI [m/s]"]])
                / 1000,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True Velocity (ECI) [km/s]", ylabels=["$v_x$", "$v_y$", "$v_z$"])
        save_figure(itm.gcf(), self.plot_dir, "velocity_ECI_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth Attitude
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
                [data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]],
                seriesLabel=f"_{trial_number}",
                linewidth=0.5
            )
        annotateMultiPlot(title="True attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
        save_figure(itm.gcf(), self.plot_dir, "attitude_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth body angular rate
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
                np.rad2deg(
                    [
                        data_dicts[i]["omega_x [rad/s]"],
                        data_dicts[i]["omega_y [rad/s]"],
                        data_dicts[i]["omega_z [rad/s]"],
                    ]
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True body angular rate [deg/s]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth magnetic field in ECI
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
                [data_dicts[i]["xMag ECI [T]"], data_dicts[i]["yMag ECI [T]"], data_dicts[i]["zMag ECI [T]"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="ECI Magnetic Field [T]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "mag_field_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth sun direction in ECI
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
                [data_dicts[i]["rSun_x ECI [m]"], data_dicts[i]["rSun_y ECI [m]"], data_dicts[i]["rSun_z ECI [m]"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Sun Position in ECI [m]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "ECI_sun_direction.png", self.close_after_saving)
        
        # ==========================================================================
        # Plot the angle between the angular momentum and the sun vector with the major inertia axis
        # load inertia from config file (temp solution)
        with open(os.path.join(self.trials_dir, "../../../configs/params.yaml"), "r") as f:
            pyparams = yaml.safe_load(f)       
        pyparams["trials"]             = self.trials
        pyparams["trials_dir"]         = self.trials_dir
        pyparams["plot_dir"]           = self.plot_dir
        pyparams["close_after_saving"] = self.close_after_saving
        # ==========================================================================
        # Pointing Plots: Controller target versus real attitude
        pointing_plots(pyparams, data_dicts, filepaths)

        # ==========================================================================
        # Actuator Plots: Reaction Wheel Speed and Torque, Magnetorquer Torque
        actuator_plots(pyparams, data_dicts, filepaths)
        
        # ========================= True gyro bias plots =========================
        filepaths = self._get_files_across_trials("gyro_bias_true.bin")

        START = time.time()
        args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.rad2deg(
                    np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
                ),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True Gyro Bias [deg/s]", ylabels=["$x$", "$y$", "$z$"])
        save_figure(itm.gcf(), self.plot_dir, "gyro_bias_true.png", self.close_after_saving)

        # ==========================================================================
        # Attitude Animation
        att_animation(pyparams, data_dicts)

    def sensor_measurement_plots(self):
        # ======================= Gyro measurement plots =======================
        filepaths = self._get_files_across_trials("gyro_measurement.bin")

        START = time.time()
        args = [(filepath, 100) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
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
        filepaths = self._get_files_across_trials("sun_sensor_measurement.bin")

        START = time.time()
        args = [(filepath, 100) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.array([data_dicts[i]["x [-]"], data_dicts[i]["y [-]"], data_dicts[i]["z [-]"]]),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Measured Sun Ray in body frame", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "sun_sensor_body_measurement.png", self.close_after_saving)

        # ====================== Magnetometer measurement plots ======================
        filepaths = self._get_files_across_trials("magnetometer_measurement.bin")

        START = time.time()
        args = [(filepath, 100) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # --------------------------------------------------------------------------
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.array([data_dicts[i]["x [T]"], data_dicts[i]["y [T]"], data_dicts[i]["z [T]"]]),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Measured B field in body frame", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "magnetometer_measurement.png", self.close_after_saving)

    def _plot_state_estimate_covariance(self):
        filepaths = self._get_files_across_trials("state_covariance.bin")

        START = time.time()
        args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
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
        for i in range(3):
            itm.subplot(3, 1, i + 1)
            itm.legend(loc = 'upper right')

        itm.figure(self.gyro_bias_error_figure)
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

        save_figure(self.attitude_error_figure, self.plot_dir, "attitude_estimate_error.png", self.close_after_saving)
        save_figure(self.gyro_bias_error_figure, self.plot_dir, "gyro_bias_estimate_error.png", self.close_after_saving)

    def EKF_error_plots(self):
        filepaths = self._get_files_across_trials("EKF_error.bin")

        START = time.time()
        args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # ==========================================================================
        # Attitude error plots
        self.attitude_error_figure = itm.figure()
        final_attitude_error_norms = []
        for i, trial_number in enumerate(self.trials):
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
        save_figure(itm.gcf(), self.plot_dir, "attitude_estimate_error_hist.png", self.close_after_saving)
        # ==========================================================================
        # Gyro Bias error plots
        self.gyro_bias_error_figure = itm.figure()
        final_gyro_bias_error_norms = []
        for i, trial_number in enumerate(self.trials):
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
        save_figure(itm.gcf(), self.plot_dir, "gyro_bias_estimate_error_hist.png", self.close_after_saving)
        # --------------------------------------------------------------------------
        self._plot_state_estimate_covariance()  # show 3 sigma bounds

    def EKF_state_plots(self):
        # ======================= Estimated gyro bias =======================
        filepaths = self._get_files_across_trials("EKF_state.bin")

        START = time.time()
        args = [(filepath, self.PERCENTAGE_OF_DATA_TO_PLOT) for (_, filepath) in filepaths]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
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
        save_figure(itm.gcf(), self.plot_dir, "attitude_estimated.png", self.close_after_saving)
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
        save_figure(itm.gcf(), self.plot_dir, "gyro_bias_estimated.png", self.close_after_saving)