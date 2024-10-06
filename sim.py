from build.world.pyphysics import rk4
from build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
import numpy as np
from scipy.spatial.transform import Rotation as R
from time import time
import datetime
from simulation_manager import logger
import os

START_TIME = time()

controller_dt = 0.1
estimator_dt = 1


def controller(state_estimate):
    return np.array([0, 0, 0, 0, 0, 1])


def estimator(w):
    pass


def run(log_directory, config_path):
    logr = logger.MultiFileLogger(log_directory)

    params = SimParams()
    params.getParamsFromFileAndSample(config_path)

    init_pos_b_wrt_ECI_in_ECI = np.array([params.EARTH_RADIUS + params.init_altitute, 0, 0])  # [m]
    init_pos_b_wrt_ECI_in_ECI_normalized = init_pos_b_wrt_ECI_in_ECI / np.linalg.norm(init_pos_b_wrt_ECI_in_ECI)
    init_ECI_q_b = R.from_quat([0, 0, 0, 1])
    init_vel_b_wrt_ECI_in_ECI = params.orbital_velocity * np.cross(
        params.orbital_plane_normal, init_pos_b_wrt_ECI_in_ECI_normalized
    )  # [m/s]
    init_vel_b_wrt_ECI_in_b = init_ECI_q_b.as_matrix().T @ init_vel_b_wrt_ECI_in_ECI  # [m/s]
    init_omega_b_wrt_ECI_in_b = np.array([0, 0, np.deg2rad(3)])  # [rad/s]

    # assert the initial position vector is orthogonal to the satellite's orbital plane normal vector
    angle_between_pos_vector_and_orbital_plane_normal = np.arccos(
        params.orbital_plane_normal.dot(init_pos_b_wrt_ECI_in_ECI_normalized)
    )
    assert abs(angle_between_pos_vector_and_orbital_plane_normal - np.pi / 2) < 1e-10

    true_initial_state = np.array(
        [
            *init_pos_b_wrt_ECI_in_ECI,
            *init_ECI_q_b.as_quat(),  # [x, y, z, w]
            *init_vel_b_wrt_ECI_in_b,
            *init_omega_b_wrt_ECI_in_b,
        ]
    )
    true_state = true_initial_state

    last_controller_update = 0
    last_estimator_update = 0
    last_print_time = -1e99

    current_time = 0
    while current_time <= params.MAX_TIME:
        controller_output = np.zeros((6, 1))
        if current_time >= last_controller_update + controller_dt:
            state_estimate = true_state  # TODO fix me
            controller_output = controller(state_estimate)
            last_controller_update = current_time
            # print(f"Controller update: {current_time}")
        if current_time >= last_estimator_update + estimator_dt:
            # w = get_gyro_measurement()
            # estimator(w)
            last_estimator_update = current_time
            # print(f"Estimator update: {current_time}")

        # state = propogate(state, sim_dt)
        if current_time >= last_print_time + 1000:
            print(f"Heartbeat: {current_time}")
            last_print_time = current_time

        u = np.zeros((6, 1))

        vel_body_wrt_ECI_in_body = true_state[7:10]
        ECI_R_b = R.from_quat(true_state[3:7]).as_matrix()
        vel_body_wrt_ECI_in_body = ECI_R_b @ vel_body_wrt_ECI_in_body

        true_state_modified_vel = true_state.copy()
        true_state_modified_vel[7:10] = vel_body_wrt_ECI_in_body
        logr.log_v(
            "state_true.bin",
            current_time,
            true_state_modified_vel,
            "time [s]",
            [
                *["x ECI [m]", "y ECI [m]", "z ECI [m]"],
                *["x", "y", "z", "w"],
                *["x ECI [m/s]", "y ECI [m/s]", "z ECI [m/s]"],
                *["x [rad/s]", "y [rad/s]", "z [rad/s]"],
            ],
        )

        true_state = rk4(
            true_state, u, params.InertiaTensor, params.InertiaTensorInverse, params.satellite_mass, params.dt
        )
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
    # REPO_NAME = "GNC-Simulation"

    # # "_rel" postfix indicates a relative filepath
    # repo_root_rel = "."  # path to "REPO_NAME/"

    # # paths relative to repo_root_rel:
    # montecarlo_rel = "montecarlo/"

    # # "_abs" suffix indicates an absolute filepath
    # repo_root_abs = os.path.realpath(repo_root_rel)
    # results_directory_abs = os.path.join(repo_root_abs, montecarlo_rel, "results/")
    # os.system(f"mkdir -p {results_directory_abs}")
    # # ensure repo_root_abs actually points to the REPO_NAME
    # assert os.path.basename(repo_root_abs) == REPO_NAME

    # # ensure paths exist
    # assert os.path.exists(repo_root_abs), f"Nonexistent: {repo_root_abs}"
    # assert os.path.exists(results_directory_abs), f"Nonexistent: {results_directory_abs}"

    # job_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # print(GREEN + f'job_name:{RESET} "{job_name}"')
    # job_directory_abs = os.path.join(results_directory_abs, job_name)
    # os.system(f"mkdir -p {job_directory_abs}")

    TRIAL_DIRECTORY = os.environ["TRIAL_DIRECTORY"]
    PARAMETER_FILEPATH = os.environ["PARAMETER_FILEPATH"]
    run(TRIAL_DIRECTORY, PARAMETER_FILEPATH)
