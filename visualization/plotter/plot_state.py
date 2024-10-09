from plot_helper import triPlot
from isolated_trace import itm


def plot_state(
    trial_number,
    data_dict,
    attitude_figure,
    gyro_bias_figure,
    accel_bias_figure,
    seriesLabel,
):
    itm.figure(attitude_figure)
    triPlot(
        data_dict["time [s]"],
        [data_dict["attitude x [deg]"], data_dict["attitude y [deg]"], data_dict["attitude z [deg]"]],
        seriesLabel=seriesLabel,
    )

    itm.figure(gyro_bias_figure)
    triPlot(
        data_dict["time [s]"],
        [data_dict["gyro bias x [deg/hr]"], data_dict["gyro bias y [deg/hr]"], data_dict["gyro bias z [deg/hr]"]],
        seriesLabel=seriesLabel,
    )

    itm.figure(accel_bias_figure)
    triPlot(
        data_dict["time [s]"],
        [data_dict["accel bias x [m/s^2]"], data_dict["accel bias y [m/s^2]"], data_dict["accel bias z [m/s^2]"]],
        seriesLabel=seriesLabel,
    )


def plot_state_error(trial_number, data_dict, attitude_figure, gyro_bias_figure, accel_bias_figure, seriesLabel=None):
    itm.figure(attitude_figure)
    triPlot(
        data_dict["time [s]"],
        [
            data_dict["attitude error x [deg]"],
            data_dict["attitude error y [deg]"],
            data_dict["attitude error z [deg]"],
        ],
        seriesLabel=seriesLabel,
    )

    itm.figure(gyro_bias_figure)
    triPlot(
        data_dict["time [s]"],
        [
            data_dict["gyro bias error x [deg/hr]"],
            data_dict["gyro bias error y [deg/hr]"],
            data_dict["gyro bias error z [deg/hr]"],
        ],
        seriesLabel=seriesLabel,
    )

    itm.figure(accel_bias_figure)
    triPlot(
        data_dict["time [s]"],
        [
            data_dict["accel bias error x [m/s^2]"],
            data_dict["accel bias error y [m/s^2]"],
            data_dict["accel bias error z [m/s^2]"],
        ],
        seriesLabel=seriesLabel,
    )
