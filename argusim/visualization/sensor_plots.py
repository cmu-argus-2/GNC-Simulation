import numpy as np
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
    # ====================== Sun Sensor measurement plots ======================
    itm.figure()
    for i, (trial_number, _) in enumerate(filepaths):
        multiPlot(
            data_dicts[i]["Time [s]"],
            np.array([data_dicts[i]["x [-]"], data_dicts[i]["y [-]"], data_dicts[i]["z [-]"]]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Measured Sun Ray in body frame", ylabels=["x", "y", "z"])
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
    annotateMultiPlot(title="Measured B field in body frame", ylabels=["x", "y", "z"])
    save_figure(itm.gcf(), plot_dir, "magnetometer_measurement.png", close_after_saving)