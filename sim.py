from build.world.pyphysics import rk4
from build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
import numpy as np
from scipy.spatial.transform import Rotation as R
from time import time

START_TIME = time()

controller_dt = 0.1
estimator_dt = 1

current_time = 0
last_controller_update = 0
last_estimator_update = 0


def controller(state_estimate):
    return np.array([0, 0, 0, 0, 0, 1])


def estimator(w):
    pass


params = SimParams()
params.getParamsFromFileAndSample("montecarlo/configs/params.yaml")

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


last_print_time = 0
while current_time <= params.MAX_TIME:
    controller_output = np.zeros((6, 1))
    if current_time >= last_controller_update + controller_dt:
        state_estimate = true_state  # TODO fix me
        controller_output = controller(state_estimate)
        last_controller_update = current_time
        print(f"Controller update: {current_time}")
    if current_time >= last_estimator_update + estimator_dt:
        # w = get_gyro_measurement()
        # estimator(w)
        last_estimator_update = current_time
        print(f"Estimator update: {current_time}")

    # state = propogate(state, sim_dt)
    if current_time >= last_print_time + 1000:
        print(f"Heartbeat: {current_time}")
        last_print_time = current_time

    u = np.zeros((6, 1))
    state = rk4(true_state, u, params.InertiaTensor, params.InertiaTensorInverse, params.satellite_mass, params.dt)

    current_time += params.dt


print(f'Sim took {time()-START_TIME:.3f} [s] "wall-clock" to simulate {params.MAX_TIME} [s]')
