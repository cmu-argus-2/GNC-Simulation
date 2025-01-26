import numpy as np
from argusim.visualization.plotter.isolated_trace import itm
from argusim.world.math.quaternions import quatrotation
from argusim.visualization.plotter.plot_helper import (
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
            np.rad2deg(
                np.array([data_dicts[i]["x [rad/s]"], data_dicts[i]["y [rad/s]"], data_dicts[i]["z [rad/s]"]])
            ),
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
            np.array([data_dicts[i]["x [T]"], data_dicts[i]["y [T]"], data_dicts[i]["z [T]"]]),
            seriesLabel=f"_{trial_number}",
        )
    annotateMultiPlot(title="Measured B field in body frame", ylabels=["x", "y", "z"])
    save_figure(itm.gcf(), plot_dir, "magnetometer_measurement.png", close_after_saving)