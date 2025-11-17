import numpy as np
import matplotlib.pyplot as plt
from argusim.visualization.isolated_trace import itm
from argusim.world.math.quaternions import quatrotation
from argusim.visualization.plot_helper import (
    multiPlot,
    annotateMultiPlot,
    save_figure,
)

def gyro_plots(pyparams, data_dicts, filepaths):
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