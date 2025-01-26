
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
from argusim.actuators import Magnetorquer



def actuator_plots(pyparams, data_dicts, filepaths):

    # ==========================================================================
    # Reaction Wheel Speed and Torque
    num_RWs = pyparams["reaction_wheels"]["N_rw"]
    trials             = pyparams["trials"]
    trials_dir         = pyparams["trials_dir"]
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]

    # G_rw_b = np.array(pyparams["rw_orientation"]).reshape(3, num_RWs)
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        rw_speed = [data_dicts[i]["omega_RW_" + str(j) + " [rad/s]"] for j in range(num_RWs)]
        rw_speed_labels = [f"RW_{j} [rad/s]" for j in range(num_RWs)]
        torque_rw = [data_dicts[i]["T_RW_" + str(j) + " [Nm]"] for j in range(num_RWs)]
        rw_torque_labels = [f"Torque_RW_{j}" for j in range(num_RWs)]
        rw_speed_torque = rw_speed + torque_rw
        rw_speed_torque_labels = rw_speed_labels + rw_torque_labels
        multiPlot(
        data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
        rw_speed_torque,
        seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Reaction Wheel Speed and Torque", ylabels=rw_speed_torque_labels)
    save_figure(itm.gcf(), plot_dir, "rw_w_T_true.png", close_after_saving)
    # ==========================================================================
    # Magnetorquer dipole moment
    num_MTBs = pyparams["magnetorquers"]["N_mtb"]
    Magnetorquers = [Magnetorquer(pyparams["magnetorquers"], IdMtb) for IdMtb in range(num_MTBs)]
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
        mtb_dipole_moment = np.zeros((num_MTBs, len(data_dicts[i]["Time [s]"])))
        for j in range(len(data_dicts[i]["Time [s]"])):
            for k in range(num_MTBs):
                Magnetorquers[k].set_voltage(volt_magnetorquer[k][j])
                mtb_dipole_moment[k][j] = np.linalg.norm(Magnetorquers[k].get_dipole_moment())

        mtb_dipole_moment_labels = [f"MTB_{j} [Cm]" for j in range(num_MTBs)]
        multiPlot(
        data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
        mtb_dipole_moment,
        seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Magnetorquer Dipole Moment [C*m]", ylabels=mtb_dipole_moment_labels)
    save_figure(itm.gcf(), plot_dir, "mtb_dipole_moment_true.png", close_after_saving)

    # ==========================================================================
    # Total Body frame torque of magnetorquers

    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
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
        data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
        total_torque,
        seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(
        title="Total Magnetorquer Body Frame Torque [Nm]", ylabels=["T_x [Nm]", "T_y [Nm]", "T_z [Nm]"]
    )
    save_figure(itm.gcf(), plot_dir, "total_mtb_body_frame_torque.png", close_after_saving)