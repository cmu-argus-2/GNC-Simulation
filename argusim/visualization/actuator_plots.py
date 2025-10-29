
import numpy as np
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
import argusim.build.actuators.pymagnetorquers as mtb


def actuator_plots(pyparams, data_dicts, filepaths):
    trials             = pyparams["trials"]
    trials_dir         = pyparams["trials_dir"]
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ==========================================================================
    # Reaction Wheel Speed and Torque
    num_RWs = pyparams["reaction_wheels"]["N_rw"]
    if num_RWs > 0:
        # G_rw_b = np.array(pyparams["rw_orientation"]).reshape(3, num_RWs)
        itm.figure()
        for i, (trial_number, _) in enumerate(filepaths):
            rw_speed = [data_dicts[i]["omega_RW_" + str(j) + " [rad/s]"] for j in range(num_RWs)]
            rw_speed_labels = [f"RW_{j} [rad/s]" for j in range(num_RWs)]
            torque_rw = [data_dicts[i]["T_RW_" + str(j) + " [Nm]"] for j in range(num_RWs)]
            rw_torque_labels = [f"T_RW_{j} [Nm]" for j in range(num_RWs)]
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
    # Magnetorquer voltages
    num_MTBs = pyparams["magnetorquers"]["N_mtb"]
    Magnetorquers = []
    for i, (trial_number, _) in enumerate(filepaths):

        with open(os.path.join(trials_dir, f"trial{trial_number}/trial_params.yaml"), "r") as f:
            pyparams2 = yaml.safe_load(f)
        # Define Magnetorquers class
        # sample G_mtb_b and resistances 
        Magnetorquers += [mtb.Magnetorquer(
            num_MTBs,
            pyparams2["mtb_resistances"],
            pyparams["magnetorquers"]["A_cross"],
            pyparams["magnetorquers"]["N_turns"],
            pyparams["magnetorquers"]["max_voltage"],
            pyparams["magnetorquers"]["max_current_rating"],
            pyparams["magnetorquers"]["max_power"],
            np.full(num_MTBs, pyparams["magnetorquers"]["inductance"]),
            np.array(pyparams2["mag_mtb_sens"]).reshape(3, 3),
            np.array(pyparams2["mtb_orientation"]).reshape(3, num_MTBs),
            pyparams2["Ahdt"],
            pyparams2["Bhdt"],
            pyparams2["Adt"],
            pyparams2["Bdt"],
            pyparams2["mtb_working_status"],
        )]
    
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
        multiPlot(
            data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
            volt_magnetorquer,
            seriesLabel=f"_{trial_number}",
        )
    mtb_voltage_labels = [f"MTB_{j} [V]" for j in range(num_MTBs)]
    annotateMultiPlot(title="Magnetorquer Voltages [V]", ylabels=mtb_voltage_labels)
    save_figure(itm.gcf(), plot_dir, "mtb_voltage_true.png", close_after_saving)

    # ==========================================================================
    # Magnetorquer dipole moment
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        # volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
        curr_magnetorquer = np.array([data_dicts[i]["I_MTB_" + str(j) + " [A]"] for j in range(num_MTBs)])
        mtb_dipole_moment = np.zeros((num_MTBs, len(data_dicts[i]["Time [s]"])))
        mtb_dipole_momentk = np.zeros(3)
        for j in range(len(data_dicts[i]["Time [s]"])):
            for k in range(num_MTBs):
                mtb_dipole_momentk = Magnetorquers[i].getSingleDipoleMoment(k, curr_magnetorquer[k][j])
                mtb_dipole_moment[k][j] = np.linalg.norm(mtb_dipole_momentk)
                # Magnetorquers[k].set_voltage(volt_magnetorquer[k][j])
                # mtb_dipole_moment[k][j] = np.linalg.norm(Magnetorquers[k].get_dipole_moment())

        mtb_dipole_moment_labels = [f"MTB_{j} [Am^2]" for j in range(num_MTBs)]
        multiPlot(
        data_dicts[i]["Time [s]"]- data_dicts[i]["Time [s]"][0],
        mtb_dipole_moment,
        seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Magnetorquer Dipole Moment [Am^2]", ylabels=mtb_dipole_moment_labels)
    save_figure(itm.gcf(), plot_dir, "mtb_dipole_moment_true.png", close_after_saving)

    # ==========================================================================
    # Total Body frame torque of magnetorquers

    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):

        curr_magnetorquer = np.array([data_dicts[i]["I_MTB_" + str(j) + " [A]"] for j in range(num_MTBs)])
        # volt_magnetorquer = np.array([data_dicts[i]["V_MTB_" + str(j) + " [V]"] for j in range(num_MTBs)])
        mag_field = np.array(
            [data_dicts[i]["xMag ECI [T]"], data_dicts[i]["yMag ECI [T]"], data_dicts[i]["zMag ECI [T]"]]
        )
        quat = np.array([data_dicts[i]["q_w"], data_dicts[i]["q_x"], data_dicts[i]["q_y"], data_dicts[i]["q_z"]])
        torque_magnetorquer = np.zeros((3, len(data_dicts[i]["Time [s]"])))
        for j in range(len(data_dicts[i]["Time [s]"])):
            RE2b = quatrotation(quat[:, j]).T
            mag_field_loc = RE2b @ mag_field[:, j]
            # volt_magnetorquer[k][j]
            curr_magnetorquerj = np.zeros(num_MTBs)
            for k in range(num_MTBs):
                curr_magnetorquerj[k] = curr_magnetorquer[k][j]
            torque_magnetorquer[:, j] = Magnetorquers[i].getTorqueb(curr_magnetorquerj, mag_field_loc)

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

