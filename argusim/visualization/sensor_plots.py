import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from argusim.visualization.isolated_trace import itm
from argusim.world.math.quaternions import quatrotation
from argusim.visualization.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)
import allantools as atools
import spiceypy as spice


def gyro_plots(pyparams, data_dicts, state_data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ======================= Gyro measurement plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([data_dicts[i]["gyro_x [deg/s]"], data_dicts[i]["gyro_y [deg/s]"], data_dicts[i]["gyro_z [deg/s]"]]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Gyro measurement [deg/s]", ylabels=["$\Omega_x$", "$\Omega_y$", "$\Omega_z$"])
    save_figure(itm.gcf(), plot_dir, "gyro_measurement.png", close_after_saving)
    
    # Allan Variance of gyro error plot
    fig, axes = plt.subplots(3)
    for i, (trial_number, _) in enumerate(filepaths):
        tt = np.array(data_dicts[i]["Time [s]"])
        true_omega = np.rad2deg(np.array([state_data_dicts[i]["omega_x [rad/s]"], state_data_dicts[i]["omega_y [rad/s]"], state_data_dicts[i]["omega_z [rad/s]"]]))
        omega_meas = np.array([data_dicts[i]["gyro_x [deg/s]"], data_dicts[i]["gyro_y [deg/s]"], data_dicts[i]["gyro_z [deg/s]"]])

        # gyro_bias = true_omega[:,:-1] - omega_meas
        gyro_bias = true_omega - omega_meas
        r = 1 / (tt[1] - tt[0])
        (tau_outx, adevx, _, _) = atools.oadev(gyro_bias[0,:], rate=r, data_type="freq",taus="all") 
        (tau_outy, adevy, _, _) = atools.oadev(gyro_bias[1,:], rate=r, data_type="freq",taus="all")
        (tau_outz, adevz, _, _) = atools.oadev(gyro_bias[2,:], rate=r, data_type="freq",taus="all")

        axes[0].loglog(tau_outx,adevx)
        axes[1].loglog(tau_outy,adevy)
        axes[2].loglog(tau_outz,adevz)

    fig.suptitle('Allan Deviation Gyro [deg/s]')

    axes[0].set_xlim([tau_outx[0], tau_outx[-1]])
    axes[0].set_ylim([min(adevx), max(adevx)])
    axes[0].set_xlabel(r'Averaging time $\tau$ [s]')
    axes[0].set_ylabel(r'$\sigma_x(\tau) [ ^{\circ}/s]$')

    axes[1].set_xlim([tau_outy[0], tau_outy[-1]])
    axes[1].set_ylim([min(adevx), max(adevx)])
    axes[1].set_xlabel(r'Averaging time $\tau$ [s]')
    axes[1].set_ylabel(r'$\sigma_y(\tau) [ ^{\circ}/s]$')

    axes[2].set_xlim([tau_outz[0], tau_outz[-1]])
    axes[2].set_ylim([min(adevz), max(adevz)])
    axes[2].set_xlabel(r'Averaging time $\tau$ [s]')
    axes[2].set_ylabel(r'$\sigma_z(\tau) [ ^{\circ}/s]$')
    
    save_figure(fig, plot_dir, "gyro_allan_deviation.png", close_after_saving)


def sunsensor_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    num_photodiodes = pyparams["photodiodes"]["num_photodiodes"]
    # ====================== Sun Sensor measurement plots ======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        time = data_dicts[i]["Time [s]"]
        sensor_data = np.array([
            data_dicts[i][f"light_sensor_lux [lx]{j}"] for j in range(num_photodiodes)
        ])
        multiPlot(
            time,
            sensor_data,
            seriesLabel=f"_{trial_number}",
        )
    ylabels = [f"Photodiode {j}" for j in range(num_photodiodes)]
    annotateMultiPlot(title="Sun Sensor Photodiode Measurements [lx]", ylabels=ylabels)
    save_figure(itm.gcf(), plot_dir, "sun_sensor_body_measurement.png", close_after_saving)


def magsensor_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ====================== Magnetometer measurement plots ======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([data_dicts[i]["mag_x_body [muT]"], data_dicts[i]["mag_y_body [muT]"], data_dicts[i]["mag_z_body [muT]"]]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Measured B field in body frame", ylabels=["x [uT]", "y [uT]", "z [uT]"])
    save_figure(itm.gcf(), plot_dir, "magnetometer_measurement.png", close_after_saving)

def gps_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ======================= GPS measurement plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([
                data_dicts[i]["gps_posx ECEF [m]"],
                data_dicts[i]["gps_posy ECEF [m]"],
                data_dicts[i]["gps_posz ECEF [m]"],
            ]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="GPS Position in ECEF", ylabels=["X [m]", "Y [m]", "Z [m]"])
    save_figure(itm.gcf(), plot_dir, "gps_position_ecef.png", close_after_saving)

    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([
                data_dicts[i]["gps_velx ECEF [m/s]"],
                data_dicts[i]["gps_vely ECEF [m/s]"],
                data_dicts[i]["gps_velz ECEF [m/s]"],
            ]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="GPS Velocity in ECEF", ylabels=["Vx [m/s]", "Vy [m/s]", "Vz [m/s]"])
    save_figure(itm.gcf(), plot_dir, "gps_velocity_ecef.png", close_after_saving)

def rtc_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    trials_dir         = pyparams["trials_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ======================= RTC measurement plots =======================
    # Allan Variance of rtc error plot
    spice.furnsh("./../data/naif0012.tls")
    fig, axes = plt.subplots(1)
    adevlims = (np.inf, -np.inf)
    taulims = (np.inf, -np.inf)
    for i, (trial_number, _) in enumerate(filepaths):
        with open(os.path.join(trials_dir, f"trial{trial_number}/trial_params.yaml"), "r") as f:
            pyparams2 = yaml.safe_load(f)
        sim_start_time = pyparams2["sim_start_time"].strftime("%Y-%m-%dT%H:%M:%S.%f")
        j2000_start_time = spice.str2et(sim_start_time)
        # et_start_time = spiceypy.spiceypy.str2et(time)[source]

        true_time = np.array(data_dicts[i]["Time [s]"] + j2000_start_time)
        meas_time = np.array(data_dicts[i]["RTC_time [s]"])

        time_error = true_time - meas_time
        time_error = time_error - time_error[0]  # remove initial bias
        if np.all(time_error == 0):
            continue
        r = 1 / (true_time[1] - true_time[0])
        (tau_outx, adevx, _, _) = atools.oadev(time_error, rate=r, data_type="freq",taus="all")
        adevlims = (min(adevlims[0], min(adevx)), max(adevlims[1], max(adevx)))
        taulims = (min(taulims[0], tau_outx[0]), max(taulims[1], tau_outx[-1]))
        axes.loglog(tau_outx,adevx)
        axes.set_xlim([taulims[0], taulims[1]])
        axes.set_ylim([adevlims[0], adevlims[1]])

    fig.suptitle('Allan Deviation RTC [s]')

    axes.set_xlabel(r'Averaging time $\tau$ [s]')
    axes.set_ylabel(r'$\sigma_x(\tau) [s]$')

    save_figure(fig, plot_dir, "rtc_allan_deviation.png", close_after_saving)

# MTB Power
def mtb_power_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    num_MTBs = pyparams["magnetorquers"]["N_mtb"]
    # ======================= MTB Power plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([data_dicts[i][f"mtb_power [W]{j}"] for j in range(num_MTBs)]),
            seriesLabel=f"_{trial_number}",
        )
    ylabels = [f"MTB {j} Power [W]" for j in range(num_MTBs)]
    annotateMultiPlot(title="Magnetorquer Power", ylabels=ylabels)
    save_figure(itm.gcf(), plot_dir, "mtb_power.png", close_after_saving)
    
    # Plot total MTB power consumption
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        total_mtb_power = np.sum(np.array([data_dicts[i][f"mtb_power [W]{j}"] for j in range(num_MTBs)]), axis=0)
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([total_mtb_power]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Total Magnetorquer Power Consumption", ylabels=["Total Power [W]"])
    save_figure(itm.gcf(), plot_dir, "mtb_total_power.png", close_after_saving)

# Solar Power
def solar_power_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    num_panels = pyparams["solar_panels"]["num_panels"]
    # ======================= Solar Power plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([data_dicts[i][f"solar_power [W]{j}"] for j in range(num_panels)]),
            seriesLabel=f"_{trial_number}",
        )
    ylabels = [f"Solar Panel {j} Power [W]" for j in range(num_panels)]
    annotateMultiPlot(title="Solar Panel Power", ylabels=ylabels)
    save_figure(itm.gcf(), plot_dir, "solar_power.png", close_after_saving)

# Jetson Power
def jetson_power_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ======================= Jetson Power plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([data_dicts[i]["Jetson Power [W]"]]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Jetson Power", ylabels=["Jetson Power [W]"])
    save_figure(itm.gcf(), plot_dir, "jetson_power.png", close_after_saving)

# Battery SoC, Capacity, Current, Voltage, Mid Voltage, TTE, TTF, Temperature
def battery_plots(pyparams, data_dicts, filepaths):
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ======================= Battery plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([
                data_dicts[i]["Battery SoC [%]"],
                data_dicts[i]["Battery Capacity [J]"],
                data_dicts[i]["Battery Current [A]"],
                data_dicts[i]["Battery Voltage [V]"],
                data_dicts[i]["Battery Mid Voltage [V]"],
                data_dicts[i]["Battery TTE [s]"],
                data_dicts[i]["Battery TTF [s]"],
                data_dicts[i]["Battery Temperature [K]"],
            ]),
            seriesLabel=f"_{trial_number}",
        )
    ylabels = [
        "SoC [%]", "Capacity [J]", "Current [A]", "Voltage [V]",
        "Mid Voltage [V]", "TTE [s]", "TTF [s]", "Temperature [K]"
    ]
    annotateMultiPlot(title="Battery Parameters", ylabels=ylabels)
    save_figure(itm.gcf(), plot_dir, "meas_battery_parameters.png", close_after_saving)

# Star Tracker Plots
def star_tracker_plots(pyparams, data_dicts, filepaths):
    if pyparams["debugFlags"]["include_star_tracker"] is False:
        return
    plot_dir           = pyparams["plot_dir"]
    close_after_saving = pyparams["close_after_saving"]
    # ======================= Star Tracker plots =======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([
                data_dicts[i]["star_tracker_qw [-]"],
                data_dicts[i]["star_tracker_qx [-]"],
                data_dicts[i]["star_tracker_qy [-]"],
                data_dicts[i]["star_tracker_qz [-]"],
            ]),
            seriesLabel=f"_{trial_number}",
        )
    ylabels = ["qw", "qx", "qy", "qz"]
    annotateMultiPlot(title="Star Tracker Quaternion Measurements", ylabels=ylabels)
    save_figure(itm.gcf(), plot_dir, "star_tracker_quaternion.png", close_after_saving)