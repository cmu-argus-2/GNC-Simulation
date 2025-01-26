import numpy as np
import matplotlib.pyplot as plt
from argusim.visualization.plotter.isolated_trace import itm
import yaml
import os
from argusim.world.math.quaternions import quatrotation
from argusim.visualization.plotter.plot_helper import (
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
    # ==========================================================================
    # Plot the sun vector in the body frame
    # ==========================================================================
    trials             = pyparams["trials"]
    trials_dir         = pyparams["trials_dir"]
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    J_ref = np.array(pyparams["inertia"]["nominal_inertia"]).reshape((3,3))
    eigenvalues, _ = np.linalg.eig(J_ref)
    J_ref_max = np.max(eigenvalues)
    delta = np.deg2rad(15)
    target_ang_mom_norm = np.linalg.norm( J_ref_max * np.deg2rad(pyparams["tgt_ss_ang_vel"]))
    max_ang_mom = 0
    if pyparams["algorithm"] == "Lyapunov":
        figname  = "sun_vector_body.png"
        figtitle = "Sun Pointing/Spin Stabilization"
        figname2  = "Histogram of Sun Point After SS Times"
        figtitle2 = "spin_stabilize_sun_point_histogram_separate.png"
    elif pyparams["algorithm"] == "BaseNP":
        figname  = "nad_vector_body.png"
        figtitle = "Nadir Pointing/Spin Stabilization"
        figname2  = "Histogram of NAdir Point After SS Times"
        figtitle2 = "spin_stabilize_nad_point_histogram_separate.png"
    else:
        return

    spin_stabilize_times = []
    tgt_point_times = []
    itm.figure()
    for i, trial_number in enumerate(trials):
        with open(os.path.join(trials_dir, f"trial{trial_number}/trial_params.yaml"), "r") as f:
            pyparams2 = yaml.safe_load(f)
        J = np.array(pyparams2[pyparams2.index("inertia") + 1]).reshape((3,3))
        eigenvalues, eigenvectors = np.linalg.eig(J)
        idx = np.argsort(eigenvalues)
        major_axis = eigenvectors[:, idx[2]]
        if major_axis[np.argmax(np.abs(major_axis))] < 0:
            major_axis = -major_axis
        bpoint_vector = []
        ang_mom_vector = []
        ang_mom_norm_vector = []
        for j in range(len(data_dicts[i]["Time [s]"])):
            # Assuming the attitude quaternion is in the form [w, x, y, z]
            quat = np.array(
                [data_dicts[i]["q_w"][j], data_dicts[i]["q_x"][j], data_dicts[i]["q_y"][j], data_dicts[i]["q_z"][j]]
            )
            # Rotate the sun vector from ECI to body frame using the quaternion
            Re2b = quatrotation(quat).T
            sun_vector_eci = np.array(
                [
                    data_dicts[i]["rSun_x ECI [m]"][j],
                    data_dicts[i]["rSun_y ECI [m]"][j],
                    data_dicts[i]["rSun_z ECI [m]"][j],
                ]
            )
            sun_vector_body = Re2b @ sun_vector_eci
            if pyparams["algorithm"] == "Lyapunov":
                tgt_vector_body = sun_vector_body / np.linalg.norm(sun_vector_body)
            
            elif pyparams["algorithm"] == "BaseNP":
                eci_pos = np.array(
                    [
                        data_dicts[i]["r_x ECI [m]"][j],
                        data_dicts[i]["r_y ECI [m]"][j],
                        data_dicts[i]["r_z ECI [m]"][j],
                    ]
                )
                zenith_vector    = eci_pos
                cross_vector     = np.array(
                    [
                        data_dicts[i]["v_x ECI [m/s]"][j], 
                        data_dicts[i]["v_y ECI [m/s]"][j], 
                        data_dicts[i]["v_z ECI [m/s]"][j]
                    ]
                )
                orbit_vector     = np.cross(zenith_vector, cross_vector)
                tgt_vector_body     = Re2b @ orbit_vector / np.linalg.norm(orbit_vector)
                if np.dot(tgt_vector_body,sun_vector_body) < 0:
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
            ang_mom = ang_mom / ang_mom_norm
            angle_am = np.rad2deg(np.arccos(np.dot(ang_mom, major_axis)))
            ang_mom_vector.append(angle_am)
            # angle_sv = np.rad2deg(np.arccos(np.dot(sun_vector_body, major_axis)))
            angle_sv = np.rad2deg(np.arccos(np.dot(tgt_vector_body, ang_mom)))
            bpoint_vector.append(angle_sv)
            
            if max(ang_mom_norm_vector) > max_ang_mom:
                max_ang_mom = max(ang_mom_norm_vector)
            
        bpoint_vector    = np.array(bpoint_vector).T
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
            [bpoint_vector, ang_mom_vector, ang_mom_norm_vector],
            seriesLabel=f"_{trial_number}",
        )

        # annotateMultiPlot(title="Sun Pointing/Spin Stabilization", 
        #                   ylabels=["SunVector/AngMom Angle [deg]", "AngMom/MajorAx Angle [deg]", "AngMom norm [Nms]"],)

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

    annotateMultiPlot(title=figtitle, 
                    ylabels=["$\\angle_{\\mathbf{s}/\\mathbf{h}} [\\degree]$", 
                            "$\\angle_{\\mathbf{h}/\\mathbf{I_{max}}} [\\degree]$",
                            "$||\\mathbf{h}|| [Nms]$"])

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
    itm.axhline(y=target_ang_mom_norm*(1-delta), color='red', linestyle='--', linewidth=1.0)
    itm.axhline(y=target_ang_mom_norm*(1+delta), color='red', linestyle='--', linewidth=1.0)
    plt.xlim([0, time_data[-1]])
    plt.ylim([0, max_ang_mom])
    plt.xlabel(time_label)

    save_figure(itm.gcf(), plot_dir, figname, close_after_saving)
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
    plt.title(figtitle2)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, figname2))
    plt.close()

    # ==========================================================================
    # Nadir Pointing
    # Orbit Pointing
    num_RWs = pyparams["reaction_wheels"]["N_rw"]
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