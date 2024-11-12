# Main entry point for each trial in a Python Job

from build.world.pyphysics import rk4
from build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
import numpy as np
from scipy.spatial.transform import Rotation as R
from time import time
from simulation_manager import logger
import os
from sensors.Sensor import SensorNoiseParams, TriAxisSensor
from sensors.SunSensor import SunSensor
from sensors.Bias import BiasParams
from algs.Estimators import Attitude_EKF

START_TIME = time()

# TODO read these in from parameter file; don't hardcode
CONTROLLER_DT = 0.1  # [s]
GYRO_DT = 0.05  # [s]
SUN_SENSOR_DT = 10  # [s]


def controller(state_estimate):
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def run(log_directory, config_path):
    logr = logger.MultiFileLogger(log_directory)

    params = SimParams(config_path)

    true_state = np.array(params.initial_true_state)  # Get initial state
    print(true_state)
    num_RWs = params.num_RWs
    num_MTBs = params.num_MTBs

    state_estimate = true_state
    initial_ECI_R_b_estimate = R.from_quat([*state_estimate[7:10], state_estimate[6]])  # TODO inverse?
    initial_gyro_bias_estimate = np.deg2rad(np.zeros(3))  # [rad/s]

    sigma_initial_attitude = np.deg2rad(5)  # [rad]
    sigma_initial_gyro_bias = np.deg2rad(5)  # [rad/s]
    sigma_gyro_white = np.deg2rad(1.5 / np.sqrt(60))  # [rad/sqrt(s)]
    sigma_gyro_bias_deriv = np.deg2rad(0.15)  # [(rad/s)/sqrt(s))]
    sigma_sunsensor = np.deg2rad(5)  # [rad]

    # TODO read these in from parameter file; don't hardcode
    initial_bias_range = np.deg2rad([0, 0])  # [-5.0, 5.0])  # [rad/s]
    sigma_w_range = np.deg2rad([0, 0])  # [0.5 / np.sqrt(60), 5.0 / np.sqrt(60)])  # [rad/sqrt(s)]
    sigma_v_range = np.deg2rad([0, 0])  # [0.05 / np.sqrt(60), 0.5 / np.sqrt(60)])  # [(rad/s)/sqrt(s))]
    scale_factor_error_range = [0, 0]  # [-0.01, 0.01]  # [-]

    gyro_params = []
    for i in range(3):
        biasParams = BiasParams.get_random_params(initial_bias_range, sigma_w_range)
        gyro_params.append(SensorNoiseParams.get_random_params(biasParams, sigma_v_range, scale_factor_error_range))
    gyro = TriAxisSensor(GYRO_DT, gyro_params)

    sunSensor = SunSensor(sigma_sunsensor)

    attitude_ekf = Attitude_EKF(
        initial_ECI_R_b_estimate,
        initial_gyro_bias_estimate,
        sigma_initial_attitude,
        sigma_initial_gyro_bias,
        sigma_gyro_white,
        sigma_gyro_bias_deriv,
        sigma_sunsensor,
        GYRO_DT,
    )

    # Logging Legend
    true_state_labels = (
        [f"r_{axis} ECI [m]" for axis in "xyz"]
        + [f"v_{axis} ECI [m/s]" for axis in "xyz"]
        + [f"q_{axis}" for axis in "wxyz"]
        + [f"omega_{axis} [rad/s]" for axis in "xyz"]
        + [f"omega_RW_{i} [rad/s]" for i in range(num_RWs)]
    )

    # measurement_labels = state_labels # TODO : Fix based on partial state measurement
    estimated_state_labels = (
        [f"r_hat_{axis} ECI [m]" for axis in "xyz"]
        + [f"v_hat_{axis} ECI [m/s]" for axis in "xyz"]
        + [f"q_hat_{axis}" for axis in "wxyz"]
        + [f"omega_hat_{axis} [rad/s]" for axis in "xyz"]
        + [f"omega_hat_RW_{i} [rad/s]" for i in range(num_RWs)]
    )

    input_labels = [f"V_MTB_{i} [V]" for i in range(num_MTBs)] + [f"V_RW_{i} [V]" for i in range(num_RWs)]

    attitude_estimate_error_labels = [f"{axis} [rad]" for axis in "xyz"]
    gyro_bias_error_labels = [f"{axis} [rad/s]" for axis in "xyz"]
    true_gyro_bias_labels = [f"{axis} [rad/s]" for axis in "xyz"]
    estimated_gyro_bias_labels = [f"{axis} [rad/s]" for axis in "xyz"]

    last_controller_update = 0
    last_gyro_measurement_time = 0
    last_sun_sensor_measurement_time = 0
    last_print_time = -1e99

    current_time = 0
    controller_command = np.zeros((num_MTBs + num_RWs,))
    while current_time <= params.MAX_TIME:
        true_ECI_R_body = R.from_quat([*true_state[7:10], true_state[6]])

        # Update Controller Command based on Controller Update Frequency
        if current_time >= last_controller_update + CONTROLLER_DT:
            controller_command = controller(state_estimate)
            assert len(controller_command) == (num_RWs + num_MTBs)
            last_controller_update = current_time

        # Sun Sensor update
        SUN_IN_VIEW = False  # TODO actually check if sun is in view
        if SUN_IN_VIEW and (current_time >= last_sun_sensor_measurement_time + SUN_SENSOR_DT):
            true_sun_ray_ECI = np.array([1, 0, 0])  # TODO get actual sun ray from cpp
            true_sun_ray_body = true_ECI_R_body.inv().as_matrix() @ true_sun_ray_ECI
            measured_sun_ray_in_body = sunSensor.get_measurement(true_sun_ray_body)

            attitude_ekf.sun_sensor_update(measured_sun_ray_in_body, true_sun_ray_ECI)

            last_sun_sensor_measurement_time = current_time

        # Propogate on Gyro
        if current_time >= last_gyro_measurement_time + GYRO_DT:
            true_omega_body_wrt_ECI_in_body = true_state[10:13]
            gyro_measurement = gyro.update(true_omega_body_wrt_ECI_in_body)
            attitude_ekf.gyro_update(gyro_measurement, current_time)
            last_gyro_measurement_time = current_time

            logr.log_v(
                "gyro_measurement.bin",
                [current_time] + gyro_measurement.tolist(),
                ["Time [s]"] + [f"{axis} [rad/s]" for axis in "xyz"],
            )

        # write the EKF's updated attitude estimate into the overall state vector
        state_estimate[6:10] = attitude_ekf.get_quat_ECI_R_b()  # [w, x, y, z]

        true_state = rk4(true_state, controller_command, params, current_time, params.dt)

        # get attitude estimate of the body wrt ECI
        estimated_ECI_R_body = attitude_ekf.get_ECI_R_b()
        attitude_estimate_error = (true_ECI_R_body * estimated_ECI_R_body.inv()).as_rotvec()

        true_gyro_bias = gyro.get_bias()
        estimated_gyro_bias = attitude_ekf.get_gyro_bias()
        gyro_bias_error = true_gyro_bias - estimated_gyro_bias

        logr.log_v(
            "attitude_ekf_error.bin",
            [current_time] + attitude_estimate_error.tolist() + gyro_bias_error.tolist(),
            ["Time [s]"] + attitude_estimate_error_labels + gyro_bias_error_labels,
        )
        logr.log_v("gyro_bias_true.bin", [current_time] + true_gyro_bias.tolist(), ["Time [s]"] + true_gyro_bias_labels)
        logr.log_v(
            "gyro_bias_estimated.bin",
            [current_time] + estimated_gyro_bias.tolist(),
            ["Time [s]"] + estimated_gyro_bias_labels,
        )

        logr.log_v(
            "states.bin",
            [current_time] + true_state.tolist() + state_estimate.tolist() + controller_command.tolist(),
            ["Time [s]"] + true_state_labels + estimated_state_labels + input_labels,
        )

        if current_time >= last_print_time + 1000:
            print(f"Heartbeat: {current_time}")
            last_print_time = current_time

        current_time += params.dt

    elapsed_seconds_wall_clock = time() - START_TIME
    speed_up = params.MAX_TIME / elapsed_seconds_wall_clock
    print(
        f'Sim ran {speed_up:.4g}x faster than realtime. Took {elapsed_seconds_wall_clock:.1f} [s] "wall-clock" to simulate {params.MAX_TIME} [s]'
    )


# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes

if __name__ == "__main__":
    TRIAL_DIRECTORY = os.environ["TRIAL_DIRECTORY"]
    PARAMETER_FILEPATH = os.environ["PARAMETER_FILEPATH"]
    run(TRIAL_DIRECTORY, PARAMETER_FILEPATH)
