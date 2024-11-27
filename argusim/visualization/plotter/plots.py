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
from argusim.build.world.pyphysics import ECI2GEOD
from argusim.world.math.quaternions import quatrotation
from argusim.actuators import Magnetorquer
import yaml
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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
                lon[k], lat[k], _ = ECI2GEOD([x_km[k]*1000, y_km[k]*1000, z_km[k]*1000], time_vec[k])
                
            # https://matplotlib.org/basemap/stable/users/examples.html
            itm.figure(ground_track)
            m.scatter(lon, lat, s=0.5, c = 'y', marker=".", label=f"_{trial_number}", latlon=True)
            m.scatter(lon[0], lat[0], marker="*", color="green", label="Start")
            m.scatter(lon[-1], lat[-1], marker="*", color="red", label="End")

        itm.figure(XYZt)
        annotateMultiPlot(title="True Position (ECI) [km]", ylabels=["r_x", "r_y", "r_z"])
        #save_figure(XYZt, self.plot_dir, "position_ECI_true.png", self.close_after_saving)

        itm.figure(ground_track)
        itm.gca().set_aspect("equal")
        itm.title("Ground Track [Green: Start    Red: End]")
        #save_figure(ground_track, self.plot_dir, "ground_track.png", self.close_after_saving)
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
        with open(os.path.join(self.trials_dir, '../../../configs/params.yaml'), "r") as f:
            pyparams = yaml.safe_load(f)
        
        J = np.array(pyparams["inertia"]["nominal_inertia"]).reshape((3,3))
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
        G_rw_b        = np.array(pyparams["reaction_wheels"]["rw_orientation"]).reshape(3, num_RWs)
        nadir_cam_dir = np.array(pyparams["nadir_cam_dir"])
        itm.figure()
        for i, trial_number in enumerate(self.trials):
            nadir_cam_dir_angle = []
            rw_orb_dir_angle = []
            for j in range(len(data_dicts[i]["Time [s]"])):
                quat = np.array([data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j]])
                RE2b = quatrotation(quat).T
                eci_pos = np.array([data_dicts[i]["r_x ECI [m]"][j], data_dicts[i]["r_y ECI [m]"][j], data_dicts[i]["r_z ECI [m]"][j]])
                nadir_vector = -RE2b @ eci_pos
                nadir_vector = nadir_vector / np.linalg.norm(nadir_vector)
                cam_angle = np.rad2deg(np.arccos(np.dot(nadir_cam_dir, nadir_vector)))
                nadir_cam_dir_angle.append(cam_angle)

                eci_vel = np.array([data_dicts[i]["v_x ECI [m/s]"][j], data_dicts[i]["v_y ECI [m/s]"][j], data_dicts[i]["v_z ECI [m/s]"][j]])
                orb_ang_dir = np.cross(eci_pos, eci_vel)
                orb_ang_dir = orb_ang_dir / np.linalg.norm(orb_ang_dir)
                orb_ang_dir = Re2b @ orb_ang_dir
                sun_pos = np.array([data_dicts[i]["rSun_x ECI [m]"][j], data_dicts[i]["rSun_y ECI [m]"][j], data_dicts[i]["rSun_z ECI [m]"][j]])
                if np.dot(sun_pos, orb_ang_dir) < 0:
                    orb_ang_dir = -orb_ang_dir
                orb_angle = np.rad2deg(np.arccos(np.dot(orb_ang_dir, G_rw_b)))
                rw_orb_dir_angle.append(orb_angle)

            multiPlot(
            data_dicts[i]["Time [s]"],
            [nadir_cam_dir_angle, rw_orb_dir_angle],
            seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(title="Nadir and Orbit Ang Mom alignment", ylabels=["Nadir Cam Dir Angle [deg]", "Orbit Dir Angle [deg]"])
        save_figure(itm.gcf(), self.plot_dir, "nad_orb_point_true.png", self.close_after_saving)
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

        # ==========================================================================
        # TODO: Video of attitude
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Attitude Animation in ECI with Sun Vector')

        quiver_nadir = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='r', label='Nadir')
        quiver_mag   = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='g', label='Mag Field')
        quiver_sp    = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='b', label='Solar Panels/+Z')
        # quiver_x2 = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, label='Min J Axis')
        # quiver_y2 = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, label='Med J Axis')
        quiver_im = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='k', label='Max J Axis')
        sun_quiver = ax.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True, color='y', label='Sun Vector')
        ax.legend([quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver], ['Nadir', 'Mag Field', 'Solar Panels/+Z', 'Max J Axis', 'Sun Vector'])

        def update_quiver(num, data_dicts, quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver):
            quat = np.array([data_dicts[0]["q_w"][num], data_dicts[0]["q_x"][num], data_dicts[0]["q_y"][num], data_dicts[0]["q_z"][num]])
            
            Re2b = quatrotation(quat)
            nadir = -np.array([data_dicts[0]["r_x ECI [m]"][num], data_dicts[0]["r_y ECI [m]"][num], data_dicts[0]["r_z ECI [m]"][num]])
            nadir = nadir / np.linalg.norm(nadir)
            mag_field = np.array([data_dicts[0]["xMag ECI [T]"][num], data_dicts[0]["yMag ECI [T]"][num], data_dicts[0]["zMag ECI [T]"][num]])
            mag_field = mag_field / np.linalg.norm(mag_field)
            body_z = Re2b @ G_rw_b.flatten()
            # inertia_min = Re2b @ eigenvectors[:, idx[0]]
            # inertia_med = Re2b @ eigenvectors[:, idx[1]]
            inertia_max = Re2b @ major_axis
            sun_vector_eci = np.array([data_dicts[0]["rSun_x ECI [m]"][num], data_dicts[0]["rSun_y ECI [m]"][num], data_dicts[0]["rSun_z ECI [m]"][num]])
            sun_vector_eci = sun_vector_eci / np.linalg.norm(sun_vector_eci)

            quiver_nadir.set_segments([[[0, 0, 0], nadir]])
            quiver_mag.set_segments([[[0, 0, 0], mag_field]])
            quiver_sp.set_segments([[[0, 0, 0], body_z]])
            quiver_im.set_segments([[[0, 0, 0], inertia_max]])
            sun_quiver.set_segments([[[0, 0, 0], sun_vector_eci]])

            return quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver
        
        fps = 24
        max_duration = 15  # seconds
        max_frames = fps * max_duration
        total_frames = len(data_dicts[0]["Time [s]"])
        step = max(1, total_frames // max_frames)

        ani = FuncAnimation(fig, update_quiver, frames=range(0, total_frames, step), fargs=(data_dicts, quiver_nadir, quiver_mag, quiver_sp, quiver_im, sun_quiver), interval=1000/fps, blit=False)
        ani.save(os.path.join(self.plot_dir, 'att_anim_BF_Sun.gif'), writer='pillow', fps=20)
        """