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
    return np.array([0, 0, 0, 0, 0, 0, 0])


def estimator(w):
    pass


def run(log_directory, config_path):
    logr = logger.MultiFileLogger(log_directory)

    params = SimParams()
    params.getParamsFromFileAndSample(config_path)

    initial_state = np.array(params.initial_state) # Get initial state
    num_RWs = params.num_RWs
    num_MTBs = params.num_MTBs

    true_state = initial_state

    last_controller_update = 0
    last_estimator_update = 0
    last_print_time = -1e99

    current_time = 0
    while current_time <= params.MAX_TIME:
        
        # Get controller Command
        controller_output = np.zeros((num_MTBs + num_RWs, 1))

        # Update Controller Command based on Controller Update Frequency
        if current_time >= last_controller_update + controller_dt:
            state_estimate = true_state  # TODO fix me
            controller_output = controller(state_estimate)
            assert(len(controller_output) == (num_RWs + num_MTBs))
            last_controller_update = current_time

        # Update Estimator Prediction at Estimator Update Frequency
        if current_time >= last_estimator_update + estimator_dt:
            # w = get_gyro_measurement()
            # estimator(w)
            last_estimator_update = current_time
            # print(f"Estimator update: {current_time}")

        if current_time >= last_print_time + 1000:
            print(f"Heartbeat: {current_time}")
            last_print_time = current_time

        logr.log_v( # TODO : Log Control Input and State Estimate at each time
            "state_true.bin",
            current_time,
            true_state,
            "time [s]",
            ([
                *["x ECI [m]", "y ECI [m]", "z ECI [m]"],
                *["x ECI [m/s]", "y ECI [m/s]", "z ECI [m/s]"],
                *["x", "y", "z", "w"],
                *["x [rad/s]", "y [rad/s]", "z [rad/s]"],
            ] + ["x [rad/s]"]*num_RWs),
        )

        true_state = rk4(true_state, controller_output, params, current_time, params.dt)
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
