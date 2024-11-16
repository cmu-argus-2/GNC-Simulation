from time import perf_counter
import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from typing import Any

import brahe
from brahe.epoch import Epoch
from brahe.constants import R_EARTH, GM_EARTH

from world.math.time import increment_epoch
from world.math.transforms import update_brahe_data_files
from world.physics.orbital_dynamics import f
from sensors.mock_vision_model2 import Camera
from nonlinear_least_squares_od import OrbitDetermination


def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 90 * 60  # s, roughly 1 orbit

    return config


def get_measurement_info(cubesat_position: np.ndarray, camera: Camera, N: int = 1) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all the information needed to represent several landmark bearing measurements.
    The number of landmark bearing measurements, M, will be some number less than or equal to N.

    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :param camera: The camera object.
    :param N: The number of landmark bearing measurements to attempt to generate.
    :return: A tuple containing a numpy array of shape (M, 3, 3) containing the rotation matrices from the body frame to the ECI frame,
             a numpy array of shape (M, 2) containing the pixel coordinates of the landmarks,
             and a numpy array of shape (M, 3) containing the landmark positions in ECI coordinates.
    """
    # define nadir cubesat attitude
    y_axis = [0, -1, 0]  # along orbital angular momentum
    z_axis = -cubesat_position / np.linalg.norm(cubesat_position)  # along radial vector
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis])

    # run vision model
    bearing_unit_vectors = camera.sample_bearing_unit_vectors(N)
    valid_intersections, landmark_positions_eci = camera.get_earth_intersections(bearing_unit_vectors, cubesat_position,
                                                                                 R_body_to_eci)
    bearing_unit_vectors = bearing_unit_vectors[valid_intersections, :]

    return np.tile(R_body_to_eci, (bearing_unit_vectors.shape[0], 1, 1)), bearing_unit_vectors, landmark_positions_eci


def is_over_daytime(epoch: Epoch, cubesat_position: np.ndarray) -> bool:
    """
    Determine if the satellite is above a portion of the Earth that is in daylight.

    :param epoch: The epoch as an instance of brahe's Epoch class.
    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: True if the satellite is above the daylight portion of the Earth, False otherwise.
    """
    return np.dot(brahe.ephemerides.sun_position(epoch), cubesat_position) > 0


def test_od():
    update_brahe_data_files()
    config = load_config()

    # set up camera, vision model, and orbit determination objects
    camera_params = config["satellite"]["camera"]
    R_body_to_camera = Rotation.from_quat(np.asarray(camera_params["orientation_in_cubesat_frame"]),
                                          scalar_first=True).as_matrix()
    camera = Camera(
        fov=np.deg2rad(120),
        R_body_to_camera=R_body_to_camera,
        t_body_to_camera=np.asarray(camera_params["position_in_cubesat_frame"])
    )
    od = OrbitDetermination(camera, dt=1 / config["solver"]["world_update_rate"])

    # set up initial state
    starting_epoch = Epoch(
        *brahe.time.mjd_to_caldate(config["mission"]["start_date"])
    )
    N = int(np.ceil(config["mission"]["duration"] * config["solver"]["world_update_rate"]))
    states = np.zeros((N, 6))
    a = R_EARTH + 600e3
    v = np.sqrt(GM_EARTH / a)
    states[0, :] = np.array([a, 0, 0, 0, 0, -v])  # polar orbit in x-z plane, angular momentum in +y direction
    epoch = starting_epoch

    # set up arrays to store measurements
    times = np.array([], dtype=int)
    Rs_body_to_eci = np.zeros(shape=(0, 3, 3))
    bearing_unit_vectors = np.zeros(shape=(0, 3))
    landmarks = np.zeros(shape=(0, 3))

    def take_measurement(t_idx: int) -> None:
        """
        Take a set of measurements at the given time index.
        Reads from the states, epoch, and mock_vision_model variables in the outer scope.
        Appends to the times, Rs_body_to_eci, pixel_coordinates, and landmarks arrays in the outer scope.

        :param t_idx: The time index at which to take the measurements.
        """
        nonlocal times, Rs_body_to_eci, bearing_unit_vectors, landmarks
        measurement_cubesat_attitudes, measurement_bearing_unit_vectors, measurement_landmarks = \
            get_measurement_info(states[t_idx, :3], camera)
        times = np.concatenate((times, np.repeat(t_idx, measurement_cubesat_attitudes.shape[0])))
        Rs_body_to_eci = np.concatenate((Rs_body_to_eci, measurement_cubesat_attitudes), axis=0)
        bearing_unit_vectors = np.concatenate((bearing_unit_vectors, measurement_bearing_unit_vectors), axis=0)
        landmarks = np.concatenate((landmarks, measurement_landmarks), axis=0)

    for t in range(0, N - 1):
        states[t + 1, :] = f(states[t, :], od.dt)

        if t % 5 == 0 and is_over_daytime(epoch, states[t, :3]):  # take a set of measurements every 5 minutes
            take_measurement(t)

        epoch = increment_epoch(epoch, 1 / config["solver"]["world_update_rate"])

    if len(times) == 0:
        raise ValueError("No measurements taken")

    start_time = perf_counter()
    estimated_states = od.fit_orbit(times, landmarks, bearing_unit_vectors, Rs_body_to_eci, N)
    print(f"Elapsed time: {perf_counter() - start_time:.2f} s")

    position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    rms_position_error = np.sqrt(np.mean(position_errors ** 2))
    print(f"RMS position error: {rms_position_error}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label="True orbit")
    ax.plot(estimated_states[:, 0], estimated_states[:, 1], estimated_states[:, 2], label="Estimated orbit")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_od()
