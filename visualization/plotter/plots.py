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
import pymap3d as pm
import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from world.math.quaternions import quatrotation
# temp (remove later)
import yaml
from actuators.magnetorquer import Magnetorquer


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

            x_km = data_dicts[i]["r_x ECI [m]"] / 1000
            y_km = data_dicts[i]["r_y ECI [m]"] / 1000
            z_km = data_dicts[i]["r_z ECI [m]"] / 1000
            multiPlot(
                data_dicts[i]["Time [s]"],
                [x_km, y_km, z_km],
                seriesLabel=f"_{trial_number}",
            )

            # TODO convert from ECI to ECEF
            tt = [datetime.datetime.utcfromtimestamp(t) for t in data_dicts[i]["Time [s]"]]
            lat, lon, _ = pm.eci2geodetic(x_km*1e3, y_km*1e3, z_km*1e3, tt)
            # lon = np.rad2deg(np.arctan2(y_km, x_km))
            # lat = np.rad2deg(np.arctan(z_km / np.hypot(x_km, y_km)))

            # https://matplotlib.org/basemap/stable/users/examples.html
            itm.figure(ground_track)
            m.scatter(lon, lat, s=0.1, marker=".", label=f"_{trial_number}", latlon=True)
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
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                np.array([data_dicts[i]["v_x ECI [m/s]"], data_dicts[i]["v_y ECI [m/s]"], data_dicts[i]["v_z ECI [m/s]"]])
                / 1000,
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True Velocity (ECI) [km/s]", ylabels=["$v_x$", "$v_y$", "$v_z$"])
        save_figure(itm.gcf(), self.plot_dir, "velocity_ECI_true.png", self.close_after_saving)
        # ==========================================================================
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
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            multiPlot(
                data_dicts[i]["Time [s]"],
                [data_dicts[i]["omega_x [rad/s]"], data_dicts[i]["omega_y [rad/s]"], data_dicts[i]["omega_z [rad/s]"]],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="True body angular rate [rad/s]", ylabels=["x", "y", "z"])
        save_figure(itm.gcf(), self.plot_dir, "body_omega_true.png", self.close_after_saving)
        # ==========================================================================
        # Plot the magnetic field in ECI
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
        # Plot the sun direction in ECI
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
        
        J = np.array(pyparams["inertia"]).reshape((3,3))
        eigenvalues, eigenvectors = np.linalg.eig(J)
        idx = np.argsort(eigenvalues)
        major_axis = eigenvectors[:, idx[2]]
        for i, trial_number in enumerate(self.trials):
            bsun_vector = []
            ang_mom_vector = []
            ang_mom_norm_vector = []
            for j in range(len(data_dicts[i]["Time [s]"])):
                # Assuming the attitude quaternion is in the form [w, x, y, z]
                quat = np.array([data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j]])
                sun_vector_eci = np.array([data_dicts[i]["rSun_x ECI [m]"][j], data_dicts[i]["rSun_y ECI [m]"][j], data_dicts[i]["rSun_z ECI [m]"][j]])
                # Rotate the sun vector from ECI to body frame using the quaternion
                Re2b = quatrotation(quat).T
                sun_vector_body = Re2b @ sun_vector_eci
                sun_vector_body = sun_vector_body / np.linalg.norm(sun_vector_body)
                angle_sv = np.rad2deg(np.arccos(np.dot(sun_vector_body, major_axis)))
                bsun_vector.append(angle_sv)
                ang_vel = np.array([data_dicts[i]["omega_x [rad/s]"][j], data_dicts[i]["omega_y [rad/s]"][j], data_dicts[i]["omega_z [rad/s]"][j]])
                ang_mom = J @ ang_vel
                ang_mom_norm = np.linalg.norm(ang_mom)
                ang_mom_norm_vector.append(ang_mom_norm)
                ang_mom = ang_mom / ang_mom_norm
                angle_am = np.rad2deg(np.arccos(np.dot(ang_mom, major_axis)))
                ang_mom_vector.append(angle_am)
                ang_mom_norm
            bsun_vector    = np.array(bsun_vector).T
            ang_mom_vector = np.array(ang_mom_vector).T
            itm.figure()
            multiPlot(
                data_dicts[i]["Time [s]"],
                [bsun_vector, ang_mom_vector, ang_mom_norm_vector],
                seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Sun Pointing/Spin Stabilization", 
                          ylabels=["SunVector/MajorAx Angle [deg]", 
                                   "AngMom/MajorAx Angle [deg]",
                                   "AngMom norm [Nms]"])
        save_figure(itm.gcf(), self.plot_dir, "sun_vector_body.png", self.close_after_saving)
        # ==========================================================================
        # Reaction Wheel Speed and Torque
        num_RWs = pyparams["N_rw"]
         
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
        num_MTBs = pyparams["N_mtb"]
        Magnetorquers = [Magnetorquer(pyparams, IdMtb) for IdMtb in range(num_MTBs)] 
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

        # ==========================================================================
        # Total Body frame torque of magnetorquers 

        itm.figure()
        for i, trial_number in enumerate(self.trials):
            volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
            mag_field = np.array([data_dicts[i]["xMag ECI [T]"], 
                                  data_dicts[i]["yMag ECI [T]"], 
                                  data_dicts[i]["zMag ECI [T]"]])
            quat = np.array([data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]])
            torque_magnetorquer = np.zeros((3, len(data_dicts[i]["Time [s]"])))
            for j in range(len(data_dicts[i]["Time [s]"])):
                RE2b = quatrotation(quat[:,j]).T
                mag_field_loc = RE2b @ mag_field[:,j]
                for k in range(num_MTBs):
                    Magnetorquers[k].set_voltage(volt_magnetorquer[k][j])
                torque_magnetorquer[:, j] = np.sum([Magnetorquers[k].get_torque(mag_field_loc) for k in range(num_MTBs)], axis=0)

            total_torque = torque_magnetorquer.tolist()
            multiPlot(
            data_dicts[i]["Time [s]"],
            total_torque,
            seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Total Magnetorquer Body Frame Torque [Nm]", ylabels=["T_x [Nm]", "T_y [Nm]", "T_z [Nm]"])
        save_figure(itm.gcf(), self.plot_dir, "total_mtb_body_frame_torque.png", self.close_after_saving)
        

        
 
