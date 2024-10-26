# Main entry point for each trial in a Python Job

from build.world.pyphysics import rk4
from build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
import numpy as np
from scipy.spatial.transform import Rotation as R
from time import time
from simulation_manager import logger
import os

START_TIME = time()

controller_dt = 0.1
estimator_dt = 1


def controller(state_estimate):
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def estimator(y):
    return y

def sensors(true_state): # TODO : sensors will return a shorter measurement vector and estimator will estimate the full state
    return true_state + np.array([5000, 5000, 5000, 5, 5, 5, 0.05, 0.05, 0.05, 0.05, 2, 2, 2, 2])*np.random.random()


def run(log_directory, config_path):
    logr = logger.MultiFileLogger(log_directory)

    params = SimParams(config_path)

    initial_state = np.array(params.initial_state) # Get initial state
    num_RWs = params.num_RWs
    num_MTBs = params.num_MTBs

    # Logging Legend
    state_labels = ["r_x ECI [m]", "r_y ECI [m]", "r_z ECI [m]", "v_x ECI [m/s]", "v_y ECI [m/s]", "v_z ECI [m/s]",
                "q_w", "q_x", "q_y", "q_z", "omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"] + ["omega_RW_" + str(i) + " [rad/s]" for i in range(num_RWs)]
    #measurement_labels = state_labels # TODO : Fix based on partial state measurement
    state_estimate_labels = ["r_hat_x ECI [m]", "r_hat_y ECI [m]", "r_hat_z ECI [m]", "v_hat_x ECI [m/s]", "v_hat_y ECI [m/s]", "v_hat_z ECI [m/s]",
                "q_hat_w", "q_hat_x", "q_hat_y", "q_hat_z", "omega_hat_x [rad/s]", "omega_hat_y [rad/s]", "omega_hat_z [rad/s]"] + ["omega_hat_RW_" + str(i) + " [rad/s]" for i in range(num_RWs)]
    input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(num_MTBs)] + ["V_RW_" + str(i) + " [V]" for i in range(num_RWs)]


    true_state = initial_state
    measured_state = true_state
    state_estimate = true_state
    print(true_state)

    last_controller_update = 0
    last_estimator_update = 0
    last_print_time = -1e99

    current_time = 0
    controller_command = np.zeros((num_MTBs + num_RWs, ))
    while current_time <= params.MAX_TIME:

        # Update Controller Command based on Controller Update Frequency
        if current_time >= last_controller_update + controller_dt:
            controller_command = controller(state_estimate)
            assert(len(controller_command) == (num_RWs + num_MTBs))
            last_controller_update = current_time

        # Update Estimator Prediction at Estimator Update Frequency
        if current_time >= last_estimator_update + estimator_dt:
            state_estimate = estimator(measured_state)
            last_estimator_update = current_time

        if current_time >= last_print_time + 1000:
            print(f"Heartbeat: {current_time}")
            last_print_time = current_time

        logr.log_v( # TODO : add state estimate and measurement labels
            "state_true.bin",
            [current_time] + true_state.tolist() + state_estimate.tolist() + controller_command.tolist(),
            ["Time [s]"] + state_labels + state_estimate_labels + input_labels
        )

        true_state = rk4(true_state, controller_command, params, current_time, params.dt)
        measured_state = sensors(true_state)

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
