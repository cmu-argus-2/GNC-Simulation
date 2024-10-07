import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Any

import brahe
from brahe.epoch import Epoch
from brahe.constants import R_EARTH

from world.math.time import increment_epoch
from world.math.transforms import update_brahe_data_files
from world.physics.orbital_dynamics import f
from sensors.mock_vision_model import Camera, MockVisionModel
from nonlinear_least_squares_od import OrbitDetermination


def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 3 orbits

    return config


def get_measurement_info(epoch: Epoch, state: np.ndarray, mock_vision_model: MockVisionModel) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all the information needed to represent several landmark bearing measurements.
    The number of landmark bearing measurements, M, will be some number less than or equal to the configured maximum
    number of correspondences, mock_vision_model.N.

    :param epoch: The measurement epoch as an instance of brahe's Epoch class.
    :param state: The state of the satellite at the measurement epoch as a numpy array of shape (6,).
    :param mock_vision_model: The mock vision model object.
    :return: A tuple containing a numpy array of shape (M, 4) containing the cubesat attitudes as quaternions in ECI,
             a numpy array of shape (M, 2) containing the pixel coordinates of the landmarks,
             and a numpy array of shape (M, 3) containing the landmark positions in ECI coordinates.
    """
    R_eci_to_ecef = brahe.frames.rECItoECEF(epoch)

    # define nadir cubesat attitude
    y_axis = [0, 1, 0]  # along orbital angular momentum
    z_axis = state[:3] / np.linalg.norm(state[:3])  # along radial vector
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis]).T  # TODO: why do we need to transpose?
    cubesat_attitude = Rotation.from_matrix(R_body_to_eci).as_quat(scalar_first=True)  # in eci

    # run vision model
    pixel_coordinates, landmark_positions_ecef = mock_vision_model.get_measurement(
        cubesat_position_in_ecef=R_eci_to_ecef @ state[:3],
        cubesat_attitude_in_ecef=Rotation.from_matrix(R_eci_to_ecef @ R_body_to_eci).as_quat(scalar_first=True)
    )
    landmark_positions_eci = (R_eci_to_ecef.T @ landmark_positions_ecef.T).T
    return np.tile(cubesat_attitude, (pixel_coordinates.shape[0], 1)), pixel_coordinates, landmark_positions_eci


def test_od():
    update_brahe_data_files()
    config = load_config()

    # set up camera, vision model, and orbit determination objects
    camera_params = config["satellite"]["camera"]
    camera = Camera(
        image_width=camera_params["image_width"],
        image_height=camera_params["image_height"],
        focal_length=camera_params["focal_length"],
        position_in_cubesat_frame=np.asarray(camera_params["position_in_cubesat_frame"]),
        orientation_in_cubesat_frame=np.asarray(camera_params["orientation_in_cubesat_frame"])
    )
    mock_vision_model = MockVisionModel(camera, max_correspondences=10, earth_radius=R_EARTH)
    od = OrbitDetermination(camera, dt=1 / config["solver"]["world_update_rate"])

    # set up initial state
    starting_epoch = Epoch(
        *brahe.time.mjd_to_caldate(config["mission"]["start_date"])
    )
    N = int(np.ceil(config["mission"]["duration"] * config["solver"]["world_update_rate"]))
    states = np.zeros((N, 6))
    states[0, :] = np.array([R_EARTH + 600e3, 0, 0, 0, 0, -7.56e3])  # polar orbit in x-z plane, angular momentum in +y direction

    # set up arrays to store measurements
    times = np.array([], dtype=int)  # every minute
    cubesat_attitudes = np.zeros(shape=(0, 4))
    pixel_coordinates = np.zeros(shape=(0, 2))
    landmarks = np.zeros(shape=(0, 3))

    epoch = starting_epoch
    for t in range(0, N - 1):
        states[t + 1, :] = f(states[t, :], od.dt)

        if t % 60 == 0:  # take a set of measurements every minute, starting at the first iteration
            measurement_cubesat_attitudes, measurement_pixel_coordinates, measurement_landmarks = \
                get_measurement_info(epoch, states[t, :], mock_vision_model)
            times = np.concatenate((times, np.repeat(t, measurement_cubesat_attitudes.shape[0])))
            cubesat_attitudes = np.vstack((cubesat_attitudes, measurement_cubesat_attitudes))
            pixel_coordinates = np.vstack((pixel_coordinates, measurement_pixel_coordinates))
            landmarks = np.vstack((landmarks, measurement_landmarks))

        epoch = increment_epoch(epoch, 1 / config["solver"]["world_update_rate"])

    estimated_states = od.fit_orbit(times, landmarks, pixel_coordinates, cubesat_attitudes)
    position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    rms_position_error = np.sqrt(np.sum(position_errors ** 2))
    print(f"RMS position error: {rms_position_error}")


if __name__ == "__main__":
    test_od()
