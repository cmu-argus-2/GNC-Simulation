import numpy as np
import quaternion as qt
import matplotlib.pyplot as plt
from argusim.visualization.isolated_trace import itm
import yaml
import os
from argusim.world.math.quaternions import quatrotation
from argusim.visualization.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)

# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes


def pointing_plots(pyparams, data_dicts, filepaths):
    # spin-stabilized pointing plots
    ss_pointing_plots(pyparams, data_dicts, filepaths)
    # three-axis pointing error plots
    if pyparams["initialization"]["start_three_axis"]:
        three_axis_pointing_error_plots(pyparams, data_dicts, filepaths)


def ss_pointing_plots(pyparams, data_dicts, filepaths):
    # ==========================================================================
    # Plot the sun vector in the body frame
    # ==========================================================================
    trials             = pyparams["trials"]
    trials_dir         = pyparams["trials_dir"]
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    J_ref = np.array(pyparams["inertia"]["nominal_inertia"]).reshape((3,3))
    # compute 
    eigenvalues, _ = np.linalg.eig(J_ref)
    J_ref_max = np.max(eigenvalues)
    delta = np.deg2rad(15)
    target_ang_mom_norm = np.linalg.norm( J_ref_max * np.deg2rad(pyparams["initialization"]["tgt_ss_ang_vel"]))
    target_ang_vel_norm = pyparams["initialization"]["tgt_ss_ang_vel"]
    max_ang_mom = 0
    max_ang_vel = 0
    algorithms = [
        {
            "key": "Lyapunov",
            "figname": "sun_vector_body.png",
            "figtitle": "Sun Pointing/Spin Stabilization",
            "figname2": "spin_stabilize_sun_point_histogram_separate.png",
            "figtitle2": "Histogram of Sun Point After SS Times",
        },
        {
            "key": "BaseNP",
            "figname": "nad_vector_body.png",
            "figtitle": "Nadir Pointing/Spin Stabilization",
            "figname2": "spin_stabilize_nad_point_histogram_separate.png",
            "figtitle2": "Histogram of Nadir Point After SS Times",
        },
    ]

    for algo in algorithms:
        spin_stabilize_times = []
        tgt_point_times = []
        itm.figure()
        for i, trial_number in enumerate(trials):
            with open(os.path.join(trials_dir, f"trial{trial_number}/trial_params.yaml"), "r") as f:
                pyparams2 = yaml.safe_load(f)
            J = np.array(pyparams2["inertia"]).reshape((3,3))
            eigenvalues, eigenvectors = np.linalg.eig(J)
            idx = np.argsort(eigenvalues)
            major_axis = eigenvectors[:, idx[2]]
            if major_axis[np.argmax(np.abs(major_axis))] < 0:
                major_axis = -major_axis
            bpoint_vector = []
            ang_mom_vector = []
            ang_mom_norm_vector = []
            ang_vel_norm_vector = []
            for j in range(len(data_dicts[i]["Time [s]"])):
                quat = np.array(
                    [data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j]]
                )
                Re2b = quatrotation(quat).T
                sun_vector_eci = np.array(
                    [
                        data_dicts[i]["rSun_x ECI [m]"][j],
                        data_dicts[i]["rSun_y ECI [m]"][j],
                        data_dicts[i]["rSun_z ECI [m]"][j],
                    ]
                )
                sun_vector_body = Re2b @ sun_vector_eci
                if algo["key"] == "Lyapunov":
                    tgt_vector_body = sun_vector_body / np.linalg.norm(sun_vector_body)
                elif algo["key"] == "BaseNP":
                    eci_pos = np.array(
                        [
                            data_dicts[i]["r_x ECI [m]"][j],
                            data_dicts[i]["r_y ECI [m]"][j],
                            data_dicts[i]["r_z ECI [m]"][j],
                        ]
                    )
                    zenith_vector = eci_pos
                    cross_vector = np.array(
                        [
                            data_dicts[i]["v_x ECI [m/s]"][j],
                            data_dicts[i]["v_y ECI [m/s]"][j],
                            data_dicts[i]["v_z ECI [m/s]"][j],
                        ]
                    )
                    orbit_vector = np.cross(zenith_vector, cross_vector)
                    tgt_vector_body = Re2b @ orbit_vector / np.linalg.norm(orbit_vector)
                    if np.dot(tgt_vector_body, sun_vector_body) < 0:
                        tgt_vector_body = -tgt_vector_body

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
                ang_vel_norm_vector.append(np.rad2deg(np.linalg.norm(ang_vel)))
                ang_mom = ang_mom / ang_mom_norm
                angle_am = np.rad2deg(np.arccos(np.dot(ang_mom, major_axis)))
                ang_mom_vector.append(angle_am)
                angle_sv = np.rad2deg(np.arccos(np.dot(tgt_vector_body, ang_mom)))
                bpoint_vector.append(angle_sv)

                if max(ang_mom_norm_vector) > max_ang_mom:
                    max_ang_mom = max(ang_mom_norm_vector)
                if max(ang_vel_norm_vector) > max_ang_vel:
                    max_ang_vel = max(ang_vel_norm_vector)

            bpoint_vector = np.array(bpoint_vector).T
            ang_mom_vector = np.array(ang_mom_vector).T
            time_data = data_dicts[i]["Time [s]"] - data_dicts[i]["Time [s]"][0]
            if time_data[-1] > 4 * 24 * 3600:
                time_data /= 24 * 3600
                time_label = "Time [days]"
            elif time_data[-1] > 4 * 3600:
                time_data /= 3600
                time_label = "Time [hours]"
            elif time_data[-1] > 4 * 60:
                time_data /= 60
                time_label = "Time [minutes]"
            else:
                time_label = "Time [s]"
            am_norm_error = np.abs(target_ang_mom_norm - ang_mom_norm_vector) / target_ang_mom_norm
            ss_condition = (am_norm_error < np.deg2rad(15)) & (ang_mom_vector < 15)
            spin_stabilize_time = time_data[np.where(ss_condition)[0][0]] if np.any(ss_condition) else time_data[-1]
            tgtp_condition = (time_data > spin_stabilize_time) & (bpoint_vector <= 10.1)
            tgt_point_time = time_data[np.where(tgtp_condition)[0][0]] if np.any(tgtp_condition) else time_data[-1]

            spin_stabilize_times.append(spin_stabilize_time)
            tgt_point_times.append(tgt_point_time)

            multiPlot(
                time_data,
                [bpoint_vector, ang_mom_vector, ang_vel_norm_vector], # ang_mom_norm_vector],
                seriesLabel=f"_{trial_number}",
            )

        time_data = data_dicts[0]["Time [s]"] - data_dicts[0]["Time [s]"][0]
        if time_data[-1] > 4 * 24 * 3600:
            time_data /= 24 * 3600
            time_label = "Time [days]"
        elif time_data[-1] > 4 * 3600:
            time_data /= 3600
            time_label = "Time [hours]"
        elif time_data[-1] > 4 * 60:
            time_data /= 60
            time_label = "Time [minutes]"
        else:
            time_label = "Time [s]"

        annotateMultiPlot(title=algo["figtitle"],
                        ylabels=["$\\angle_{\\mathbf{s}/\\mathbf{h}} [\\degree]$",
                                "$\\angle_{\\mathbf{h}/\\mathbf{I_{max}}} [\\degree]$",
                                "$||\\omega|| [deg/s]$"])

        # sun pointing threshold
        itm.subplot(3, 1, 1)
        itm.axhline(y=10, color='red', linestyle='--', linewidth=1.0)
        plt.xlim([0, time_data[-1]])
        plt.ylim([0, 180])
        plt.xlabel(time_label)
        # ang mom pointing threshold
        itm.subplot(3, 1, 2)
        itm.axhline(y=15, color='red', linestyle='--', linewidth=1.0)
        plt.xlim([0, time_data[-1]])
        plt.ylim([0, 180])
        plt.xlabel(time_label)
        # ang mom norm threshold
        itm.subplot(3, 1, 3)
        itm.axhline(y=target_ang_vel_norm*(1-delta), color='red', linestyle='--', linewidth=1.0)
        itm.axhline(y=target_ang_vel_norm*(1+delta), color='red', linestyle='--', linewidth=1.0)
        plt.xlim([0, time_data[-1]])
        plt.ylim([0, max_ang_vel])
        plt.xlabel(time_label)

        save_figure(itm.gcf(), plot_dir, algo["figname"], close_after_saving)
        # ==========================================================================
        # Plot the spin stabilization time and the sun pointing after spin stabilization time separately
        plt.figure()

        # Spin stabilization time
        plt.subplot(2, 1, 1)
        plt.hist(spin_stabilize_times, bins=20, alpha=0.7, label='Spin Stabilize Time')
        plt.xlabel(time_label)
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Spin Stabilize Times')

        # Sun pointing time after spin stabilization
        plt.subplot(2, 1, 2)
        tgt_point_times_minus_spin_stabilize_times = [tgt_point_times[i] - spin_stabilize_times[i] for i in range(len(spin_stabilize_times))]
        plt.hist(tgt_point_times_minus_spin_stabilize_times, bins=20, alpha=0.7, label='Sun Point Time')
        plt.xlabel(time_label)
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(algo["figtitle2"])

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, algo["figname2"]))
        plt.close()

    # ==========================================================================
    # Nadir Pointing
    # Orbit Pointing
    num_RWs = pyparams["reaction_wheels"]["N_rw"]
    if num_RWs > 0:
        G_rw_b = np.array(pyparams["reaction_wheels"]["rw_orientation"]).reshape(3, num_RWs)
        nadir_cam_dir = np.array(pyparams["nadir_cam_dir"])
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
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
                orb_ang_dir = RE2b @ orb_ang_dir
                sun_pos = np.array(
                    [
                        data_dicts[i]["rSun_x ECI [m]"][j],
                        data_dicts[i]["rSun_y ECI [m]"][j],
                        data_dicts[i]["rSun_z ECI [m]"][j],
                    ]
                )
                sun_pos = RE2b @ sun_pos
                if np.dot(sun_pos, orb_ang_dir) < 0:
                    orb_ang_dir = -orb_ang_dir
                orb_angle = np.rad2deg(np.arccos(np.dot(orb_ang_dir, G_rw_b)))
                rw_orb_dir_angle.append(orb_angle)

            multiPlot(
            data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
            [nadir_cam_dir_angle, rw_orb_dir_angle],
            seriesLabel=f"_{trial_number}",
            )
        annotateMultiPlot(
            title="Nadir and Orbit Ang Mom alignment", ylabels=["Nadir Cam Dir Angle [deg]", "Orbit Dir Angle [deg]"]
        )
        save_figure(itm.gcf(), plot_dir, "nad_orb_point_true.png", close_after_saving)


def three_axis_pointing_error_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    trials_dir         = pyparams["trials_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ==========================================================================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        with open(os.path.join(trials_dir, f"trial{trial_number}/trial_params.yaml"), "r") as f:
            pyparams2 = yaml.safe_load(f)

        att_error_xyz = []
        for j in range(len(data_dicts[i]["Time [s]"])):
            q = np.quaternion(data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j])

            if pyparams["initialization"]["three_axis_target"] == "Inertial":
                target_quat = np.quaternion(*pyparams2["target_attitude"])
            elif pyparams["initialization"]["three_axis_target"] == "Nadir":
                r_eci = np.array([
                    data_dicts[i]["r_x ECI [m]"][j],
                    data_dicts[i]["r_y ECI [m]"][j],
                    data_dicts[i]["r_z ECI [m]"][j]
                ])
                v_eci = np.array([
                    data_dicts[i]["v_x ECI [m/s]"][j],
                    data_dicts[i]["v_y ECI [m/s]"][j],
                    data_dicts[i]["v_z ECI [m/s]"][j]
                ])
                target_quat, _ = get_nadir_states(r_eci, v_eci)
            else:
                raise ValueError("Invalid targetmode option in adcs_params.yaml")
            
            quaternion_error = q.conj() * target_quat
            # attitude_error = qt.as_rotation_vector(quaternion_error)
            attitude_error = quat2rotvector(quaternion_error)
            att_error_xyz.append(attitude_error)
        att_error_xyz = np.array(att_error_xyz).T  # shape (3, N)
        time_data = data_dicts[i]["Time [s]"] - data_dicts[i]["Time [s]"][0]
        if time_data[-1] > 4 * 24 * 3600:
            time_data /= 24 * 3600
            time_label = "Time [days]"
        elif time_data[-1] > 4 * 3600:
            time_data /= 3600
            time_label = "Time [hours]"
        elif time_data[-1] > 4 * 60:
            time_data /= 60
            time_label = "Time [minutes]"
        else:
            time_label = "Time [s]"

        multiPlot(
            time_data,
            np.rad2deg(att_error_xyz),
            seriesLabel=f"_{trial_number}",
        )
    time_data = data_dicts[0]["Time [s]"] - data_dicts[0]["Time [s]"][0]
    if time_data[-1] > 4 * 24 * 3600:
        time_data /= 24 * 3600
        time_label = "Time [days]"
    elif time_data[-1] > 4 * 3600:
        time_data /= 3600
        time_label = "Time [hours]"
    elif time_data[-1] > 4 * 60:
        time_data /= 60
        time_label = "Time [minutes]"
    else:
        time_label = "Time [s]"
    annotateMultiPlot(title="Three Axis Pointing Error [deg]", ylabels=["$x$", "$y$", "$z$"])
    save_figure(itm.gcf(), plot_dir, "three_axis_pointing_error_xyz.png", close_after_saving)

    # ==========================================================================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        omega_error_xyz = []
        for j in range(len(data_dicts[i]["Time [s]"])):
            omega = np.array([
                data_dicts[i]["omega_x [rad/s]"][j],
                data_dicts[i]["omega_y [rad/s]"][j],
                data_dicts[i]["omega_z [rad/s]"][j]
            ])
            if pyparams["initialization"]["three_axis_target"] == "Inertial":
                target_omega = np.array([0.0, 0.0, 0.0])
            elif pyparams["initialization"]["three_axis_target"] == "Nadir":
                r_eci = np.array([
                    data_dicts[i]["r_x ECI [m]"][j],
                    data_dicts[i]["r_y ECI [m]"][j],
                    data_dicts[i]["r_z ECI [m]"][j]
                ])
                v_eci = np.array([
                    data_dicts[i]["v_x ECI [m/s]"][j],
                    data_dicts[i]["v_y ECI [m/s]"][j],
                    data_dicts[i]["v_z ECI [m/s]"][j]
                ])
                _, target_omega = get_nadir_states(r_eci, v_eci)
            else:
                raise ValueError("Invalid targetmode option in adcs_params.yaml")
            omega_error = omega - target_omega
            omega_error_xyz.append(omega_error)
        omega_error_xyz = np.array(omega_error_xyz).T  # shape (3, N)
        time_data = data_dicts[i]["Time [s]"] - data_dicts[i]["Time [s]"][0]
        if time_data[-1] > 4 * 24 * 3600:
            time_data /= 24 * 3600
            time_label = "Time [days]"
        elif time_data[-1] > 4 * 3600:
            time_data /= 3600
            time_label = "Time [hours]"
        elif time_data[-1] > 4 * 60:
            time_data /= 60
            time_label = "Time [minutes]"
        else:
            time_label = "Time [s]"
        multiPlot(
            time_data,
            np.rad2deg(omega_error_xyz),
            seriesLabel=f"_{trial_number}",
        )
    time_data = data_dicts[0]["Time [s]"] - data_dicts[0]["Time [s]"][0]
    if time_data[-1] > 4 * 24 * 3600:
        time_data /= 24 * 3600
        time_label = "Time [days]"
    elif time_data[-1] > 4 * 3600:
        time_data /= 3600
        time_label = "Time [hours]"
    elif time_data[-1] > 4 * 60:
        time_data /= 60
        time_label = "Time [minutes]"
    else:
        time_label = "Time [s]"
    annotateMultiPlot(title="Three Axis Angular Velocity Error [deg/s]", ylabels=["$x$", "$y$", "$z$"])
    itm.subplot(3, 1, 1)
    plt.xlim([0, time_data[-1]])
    plt.xlabel(time_label)
    itm.subplot(3, 1, 2)
    plt.xlim([0, time_data[-1]])
    plt.xlabel(time_label)
    itm.subplot(3, 1, 3)
    plt.xlim([0, time_data[-1]])
    plt.xlabel(time_label)
    save_figure(itm.gcf(), plot_dir, "three_axis_angular_velocity_error_xyz.png", close_after_saving)

    # ==========================================================================
    # Plot norm of pointing error and norm of angular velocity error
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        with open(os.path.join(trials_dir, f"trial{trial_number}/trial_params.yaml"), "r") as f:
            pyparams2 = yaml.safe_load(f)
        # Norm of pointing error
        att_error_xyz = []
        omega_error_xyz = []
        for j in range(len(data_dicts[i]["Time [s]"])):
            q = np.quaternion(data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j])
            omega = np.array([
                data_dicts[i]["omega_x [rad/s]"][j],
                data_dicts[i]["omega_y [rad/s]"][j],
                data_dicts[i]["omega_z [rad/s]"][j]
            ])
            if pyparams["initialization"]["three_axis_target"] == "Inertial":
                target_quat = np.quaternion(*pyparams2["target_attitude"])
                target_omega = np.array([0.0, 0.0, 0.0])
            elif pyparams["initialization"]["three_axis_target"] == "Nadir":
                r_eci = np.array([
                    data_dicts[i]["r_x ECI [m]"][j],
                    data_dicts[i]["r_y ECI [m]"][j],
                    data_dicts[i]["r_z ECI [m]"][j]
                ])
                v_eci = np.array([
                    data_dicts[i]["v_x ECI [m/s]"][j],
                    data_dicts[i]["v_y ECI [m/s]"][j],
                    data_dicts[i]["v_z ECI [m/s]"][j]
                ])
                target_quat, target_omega = get_nadir_states(r_eci, v_eci)
            else:
                raise ValueError("Invalid targetmode option in adcs_params.yaml")
            quaternion_error = q.conj() * target_quat
            # attitude_error = qt.as_rotation_vector(quaternion_error)
            attitude_error = quat2rotvector(quaternion_error)
            att_error_xyz.append(attitude_error)
            omega_error = omega - target_omega
            omega_error_xyz.append(omega_error)
        att_error_xyz = np.array(att_error_xyz)  # shape (N, 3)
        att_error_norm = np.linalg.norm(att_error_xyz, axis=1)
        omega_error_xyz = np.array(omega_error_xyz)  # shape (N, 3)
        omega_error_norm = np.linalg.norm(omega_error_xyz, axis=1)
        time_data = data_dicts[i]["Time [s]"] - data_dicts[i]["Time [s]"][0]
        if time_data[-1] > 4 * 24 * 3600:
            time_data /= 24 * 3600
            time_label = "Time [days]"
        elif time_data[-1] > 4 * 3600:
            time_data /= 3600
            time_label = "Time [hours]"
        elif time_data[-1] > 4 * 60:
            time_data /= 60
            time_label = "Time [minutes]"
        else:
            time_label = "Time [s]"
        itm.subplot(2, 1, 1)
        itm.plot(time_data, np.rad2deg(att_error_norm), label=f"_{trial_number}")
        itm.ylabel("Pointing Error Norm [deg]")
        itm.subplot(2, 1, 2)
        itm.plot(time_data, np.rad2deg(omega_error_norm), label=f"_{trial_number}")
        itm.ylabel("Angular Velocity Error Norm [deg/s]")
        itm.xlabel(time_label)
    itm.subplot(2, 1, 1)
    itm.title("Norm of Pointing Error")
    itm.legend()
    itm.subplot(2, 1, 2)
    itm.title("Norm of Angular Velocity Error")
    itm.legend()
    save_figure(itm.gcf(), plot_dir, "pointing_and_angular_velocity_error_norm.png", close_after_saving)


def get_nadir_states(r_eci, v_eci):
    # return the nadir pointing attitude and angular velocity for the input data
    z = -r_eci / np.linalg.norm(r_eci)  # nadir
    y = np.cross(z, v_eci)  # negative orbit normal
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)  # ~ velocity vector
    x = x / np.linalg.norm(x)
    nadir_rotm = np.column_stack((x, y, z))
    # Mconv = np.array([[0, 1, 0],
    #                   [0, 0, 1],
    #                   [1, 0, 0]])
    Mconv = np.eye(3)
    nadir_rotm = nadir_rotm @ Mconv
    nadir_quat = qt.from_rotation_matrix(nadir_rotm)  # scalar first
    omega_eci = np.cross(r_eci, v_eci) / (np.linalg.norm(r_eci)**2)
    nadir_omega = nadir_rotm.T @ omega_eci

    return nadir_quat, nadir_omega

def quat2rotvector(quat):
    rotvec = qt.as_rotation_vector(quat)
    theta = np.linalg.norm(rotvec)
    if theta > np.pi:
        rotvec = rotvec * ((theta - 2 * np.pi * ((theta + np.pi) // (2 * np.pi))) / theta)
    
    return rotvec