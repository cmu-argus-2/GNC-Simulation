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
from world.math.quaternions import quatrotation
from actuators.magnetorquer import Magnetorquer
import yaml


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
        m.bluemarble()
        m.drawcoastlines(linewidth=0.5)
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
        # Truth Velocity (ECI)
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
        # Truth Attitude
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
        save_figure(itm.gcf(), self.plot_dir, "attitude_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth body angular rate
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
        annotateMultiPlot(title="True body angular rate [rad/s]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth magnetic field in ECI
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [data_dicts[i]["xMag ECI [T]"], data_dicts[i]["yMag ECI [T]"], data_dicts[i]["zMag ECI [T]"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="ECI Magnetic Field [T]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "mag_field_true.png", self.close_after_saving)
        # ==========================================================================
        # Truth sun direction in ECI
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [data_dicts[i]["rSun_x ECI [m]"], data_dicts[i]["rSun_y ECI [m]"], data_dicts[i]["rSun_z ECI [m]"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Sun Position in ECI [m]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "ECI_sun_direction.png", self.close_after_saving)
        # ==========================================================================
        # Plot the angle between the angular momentum and the sun vector with the major inertia axis
        # load inertia from config file (temp solution)
        with open("../../montecarlo/configs/params.yaml", "r") as f:
            pyparams = yaml.safe_load(f)

        J = np.array(pyparams["inertia"]["nominal_inertia"]).reshape((3, 3))
        eigenvalues, eigenvectors = np.linalg.eig(J)
        idx = np.argsort(eigenvalues)
        major_axis = eigenvectors[:, idx[2]]
        if major_axis[np.argmax(np.abs(major_axis))] < 0:
            major_axis = -major_axis
        for i, trial_number in enumerate(self.trials):
            bsun_vector = []
            ang_mom_vector = []
            ang_mom_norm_vector = []
            for j in range(len(data_dicts[i]["Time [s]"])):
                # Assuming the attitude quaternion is in the form [w, x, y, z]
                quat = np.array(
                    [data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j]]
                )
                sun_vector_eci = np.array(
                    [
                        data_dicts[i]["rSun_x ECI [m]"][j],
                        data_dicts[i]["rSun_y ECI [m]"][j],
                        data_dicts[i]["rSun_z ECI [m]"][j],
                    ]
                )
                # Rotate the sun vector from ECI to body frame using the quaternion
                Re2b = quatrotation(quat).T
                sun_vector_body = Re2b @ sun_vector_eci
                sun_vector_body = sun_vector_body / np.linalg.norm(sun_vector_body)
                angle_sv = np.rad2deg(np.arccos(np.dot(sun_vector_body, major_axis)))
                bsun_vector.append(angle_sv)
                ang_vel = np.array(
                    [
                        data_dicts[i]["omega_x [rad/s]"][j],
                        data_dicts[i]["omega_y [rad/s]"][j],
                        data_dicts[i]["omega_z [rad/s]"][j],
                    ]
                )
                ang_mom = J @ ang_vel
                ang_mom_norm = np.linalg.norm(ang_mom)
                ang_mom_norm_vector.append(ang_mom_norm)
                ang_mom = ang_mom / ang_mom_norm
                angle_am = np.rad2deg(np.arccos(np.dot(ang_mom, major_axis)))
                ang_mom_vector.append(angle_am)
                ang_mom_norm
            bsun_vector = np.array(bsun_vector).T
            ang_mom_vector = np.array(ang_mom_vector).T
            itm.figure()
            multiPlot(
                data_dicts[i]["Time [s]"],
                [bsun_vector, ang_mom_vector, ang_mom_norm_vector],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(
            title="Sun Pointing/Spin Stabilization",
            ylabels=["SunVector/MajorAx Angle [deg]", "AngMom/MajorAx Angle [deg]", "AngMom norm [Nms]"],
        )
        save_figure(itm.gcf(), self.plot_dir, "sun_vector_body.png", self.close_after_saving)
        # ==========================================================================
        # Reaction Wheel Speed and Torque
        num_RWs = pyparams["reaction_wheels"]["N_rw"]

        # G_rw_b = np.array(pyparams["rw_orientation"]).reshape(3, num_RWs)
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            rw_speed = [data_dicts[i]["omega_RW_" + str(j) + " [rad/s]"] for j in range(num_RWs)]
            rw_speed_labels = [f"RW_{j}" for j in range(num_RWs)]
            torque_rw = [data_dicts[i]["T_RW_" + str(j) + " [Nm]"] for j in range(num_RWs)]
            rw_torque_labels = [f"Torque_RW_{j}" for j in range(num_RWs)]
            rw_speed_torque = rw_speed + torque_rw
            rw_speed_torque_labels = rw_speed_labels + rw_torque_labels
            multiPlot(
                data_dicts[i]["Time [s]"],
                rw_speed_torque,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Reaction Wheel Speed and Torque", ylabels=rw_speed_torque_labels)
        save_figure(itm.gcf(), self.plot_dir, "rw_w_T_true.png", self.close_after_saving)
        # ==========================================================================
        # Magnetorquer dipole moment
        num_MTBs = pyparams["magnetorquers"]["N_mtb"]
        Magnetorquers = [Magnetorquer(pyparams["magnetorquers"], IdMtb) for IdMtb in range(num_MTBs)]
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
            mtb_dipole_moment = np.zeros((num_MTBs, len(data_dicts[i]["Time [s]"])))
            for j in range(len(data_dicts[i]["Time [s]"])):
                for k in range(num_MTBs):
                    Magnetorquers[k].set_voltage(volt_magnetorquer[k][j])
                    mtb_dipole_moment[k][j] = np.linalg.norm(Magnetorquers[k].get_dipole_moment())

            mtb_dipole_moment_labels = [f"MTB_{j} [Cm]" for j in range(num_MTBs)]
            multiPlot(
                data_dicts[i]["Time [s]"],
                mtb_dipole_moment,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Magnetorquer Dipole Moment [C*m]", ylabels=mtb_dipole_moment_labels)
        save_figure(itm.gcf(), self.plot_dir, "mtb_dipole_moment_true.png", self.close_after_saving)
        # ==========================================================================
        # Nadir Pointing
        # Orbit Pointing
        G_rw_b = np.array(pyparams["reaction_wheels"]["rw_orientation"]).reshape(3, num_RWs)
        nadir_cam_dir = np.array(pyparams["nadir_cam_dir"])
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            nadir_cam_dir_angle = []
            rw_orb_dir_angle = []
            for j in range(len(data_dicts[i]["Time [s]"])):
                quat = np.array(
                    [data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j]]
                )
                RE2b = quatrotation(quat).T
                eci_pos = np.array(
                    [data_dicts[i]["r_x ECI [m]"][j], data_dicts[i]["r_y ECI [m]"][j], data_dicts[i]["r_z ECI [m]"][j]]
                )
                nadir_vector = -RE2b @ eci_pos
                nadir_vector = nadir_vector / np.linalg.norm(nadir_vector)
                cam_angle = np.rad2deg(np.arccos(np.dot(nadir_cam_dir, nadir_vector)))
                nadir_cam_dir_angle.append(cam_angle)

                eci_vel = np.array(
                    [
                        data_dicts[i]["v_x ECI [m/s]"][j],
                        data_dicts[i]["v_y ECI [m/s]"][j],
                        data_dicts[i]["v_z ECI [m/s]"][j],
                    ]
                )
                orb_ang_dir = np.cross(eci_pos, eci_vel)
                orb_ang_dir = orb_ang_dir / np.linalg.norm(orb_ang_dir)
                orb_ang_dir = Re2b @ orb_ang_dir
                sun_pos = np.array(
                    [
                        data_dicts[i]["rSun_x ECI [m]"][j],
                        data_dicts[i]["rSun_y ECI [m]"][j],
                        data_dicts[i]["rSun_z ECI [m]"][j],
                    ]
                )
                if np.dot(sun_pos, orb_ang_dir) < 0:
                    orb_ang_dir = -orb_ang_dir
                orb_angle = np.rad2deg(np.arccos(np.dot(orb_ang_dir, G_rw_b)))
                rw_orb_dir_angle.append(orb_angle)

            multiPlot(
                data_dicts[i]["Time [s]"],
                [nadir_cam_dir_angle, rw_orb_dir_angle],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(
            title="Nadir and Orbit Ang Mom alignment", ylabels=["Nadir Cam Dir Angle [deg]", "Orbit Dir Angle [deg]"]
        )
        save_figure(itm.gcf(), self.plot_dir, "nad_orb_point_true.png", self.close_after_saving)
        # ==========================================================================
        # Total Body frame torque of magnetorquers

        itm.figure()
        for i, trial_number in enumerate(self.trials):
            volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
            mag_field = np.array(
                [data_dicts[i]["xMag ECI [T]"], data_dicts[i]["yMag ECI [T]"], data_dicts[i]["zMag ECI [T]"]]
            )
            quat = np.array([data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]])
            torque_magnetorquer = np.zeros((3, len(data_dicts[i]["Time [s]"])))
            for j in range(len(data_dicts[i]["Time [s]"])):
                RE2b = quatrotation(quat[:, j]).T
                mag_field_loc = RE2b @ mag_field[:, j]
                for k in range(num_MTBs):
                    Magnetorquers[k].set_voltage(volt_magnetorquer[k][j])
                torque_magnetorquer[:, j] = np.sum(
                    [Magnetorquers[k].get_torque(mag_field_loc) for k in range(num_MTBs)], axis=0
                )

            total_torque = torque_magnetorquer.tolist()
            multiPlot(
                data_dicts[i]["Time [s]"],
                total_torque,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(
            title="Total Magnetorquer Body Frame Torque [Nm]", ylabels=["T_x [Nm]", "T_y [Nm]", "T_z [Nm]"]
        )
        save_figure(itm.gcf(), self.plot_dir, "total_mtb_body_frame_torque.png", self.close_after_saving)
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

    def sensor_measurement_plots(self):
        # ======================= Gyro measurement plots =======================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/gyro_measurement.bin"))

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
        # ====================== Magnetometor measurement plots ======================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/magnetometer_measurement.bin"))

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
                np.array([data_dicts[i]["x [T]"], data_dicts[i]["y [T]"], data_dicts[i]["z [T]"]]),
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Measured B field in body frame", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "magnetometer_measurement.png", self.close_after_saving)

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

    def EKF_error_plots(self):
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/EKF_error.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
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
        plt.hist(
            final_attitude_error_norms,
            bins=np.arange(min(final_attitude_error_norms), max(final_attitude_error_norms) + 1, 1),
            cumulative=True,
            edgecolor="black",
        )
        percentile_95 = np.percentile(final_attitude_error_norms, 95)
        itm.axvline(x=percentile_95, color='red', linestyle='--', label=f'95th Percentile: {percentile_95:.2f} [deg]')
        plt.title("RSS Final Attitude Estimate Error - CDF")
        plt.xlabel("[deg]")
        plt.ylabel("%-tile")
        plt.legend()
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
        plt.hist(
            final_gyro_bias_error_norms,
            bins=np.arange(min(final_gyro_bias_error_norms), max(final_gyro_bias_error_norms) + 0.1, 0.1),
            cumulative=True,
            edgecolor="black",
        )
        percentile_95 = np.percentile(final_gyro_bias_error_norms, 95)
        itm.axvline(x=percentile_95, color='red', linestyle='--', label=f'95th Percentile: {percentile_95:.2f} [deg/s]')
        plt.title("RSS Final Gyro Bias Estimate Error - CDF")
        plt.xlabel("[deg/s]")
        plt.ylabel("%-tile")
        plt.legend()
        save_figure(itm.gcf(), self.plot_dir, "gyro_bias_estimate_error_hist.png", self.close_after_saving)
        # --------------------------------------------------------------------------
        self._plot_state_estimate_covariance()  # show 3 sigma bounds

    def EKF_state_plots(self):
        # ======================= Estimated gyro bias =======================
        filenames = []
        for trial_number in self.trials:
            filenames.append(os.path.join(self.trials_dir, f"trial{trial_number}/EKF_state.bin"))

        START = time.time()
        args = [(filename, self.PERCENTAGE_OF_DATA_TO_PLOT) for filename in filenames]
        with Pool() as pool:
            data_dicts = pool.map(parse_bin_file_wrapper, args)
        END = time.time()
        print(f"Elapsed time to read in data: {END-START:.2f} s")
        # ==========================================================================
        # Estimated Attitude
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Estimated attitude [-]", ylabels=["$q_w$", "$q_x$", "$q_y$", "$q_z$"])
        save_figure(itm.gcf(), self.plot_dir, "attitude_estimated.png", self.close_after_saving)
        # ==========================================================================
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
