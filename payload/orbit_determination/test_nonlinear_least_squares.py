import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Any

import brahe
from brahe.epoch import Epoch
from brahe.constants import R_EARTH

from world.math.time import increment_epoch
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
    Get all the information needed to represent a single landmark bearing measurement.

    :param epoch: The measurement epoch as an instance of brahe's Epoch class.
    :param state: The state of the satellite at the measurement epoch as a numpy array of shape (6,).
    :param mock_vision_model: The mock vision model object.
    :return: A tuple containing numpy arrays of the cubesat attitude as a quaternion in ECI, the pixel coordinates of the landmark, and the landmark position in ECI coordinates.
    """
    R_eci_to_ecef = brahe.frames.rECItoECEF(epoch)

    # define nadir cubesat attitude
    y_axis = [0, 1, 0]  # along orbital angular momentum
    z_axis = state[:3] / np.linalg.norm(state[:3])  # along radial vector
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis])
    cubesat_attitude = Rotation.from_matrix(R_body_to_eci).as_quat(scalar_first=True)  # in eci

    # run vision model
    correspondences = mock_vision_model.get_measurement(
        cubesat_position_in_ecef=R_eci_to_ecef @ state[:3],
        cubesat_attitude_in_ecef=Rotation.from_matrix(R_eci_to_ecef @ R_body_to_eci).as_quat(scalar_first=True)
    )
    landmark_position_eci = R_eci_to_ecef.T @ correspondences[0].ecef_coordinate
    return cubesat_attitude, correspondences[0].pixel_coordinate, landmark_position_eci


def test_od():
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
    mock_vision_model = MockVisionModel(camera, max_correspondences=1, earth_radius=R_EARTH)
    od = OrbitDetermination(camera, dt=1 / config["solver"]["world_update_rate"])

    # set up initial state
    starting_epoch = Epoch(
        *brahe.time.mjd_to_caldate(config["mission"]["start_date"])
    )
    N = int(np.ceil(config["mission"]["duration"] * config["solver"]["world_update_rate"]))
    states = np.zeros((N, 6))
    states[0] = np.array([R_EARTH + 600e3, 0, 0, 0, 0, -7.56e3])  # polar orbit in x-z plane, angular momentum in +y direction

    # set up arrays to store measurements
    times = np.arange(1, N, 60)  # every minute
    cubesat_attitudes = np.zeros((len(times), 4))
    pixel_coordinates = np.zeros((len(times), 2))
    landmarks = np.zeros((len(times), 3))
    idx = 0  # helper index for storing measurements sequentially

    epoch = starting_epoch
    for t in range(1, N):
        states[t, :] = od.f(states[t - 1, :], epoch)

        if t in times:
            cubesat_attitudes[idx, :], pixel_coordinates[idx, :], landmarks[idx, :] = \
                get_measurement_info(epoch, states[t, :], mock_vision_model)
            idx += 1

        epoch = increment_epoch(epoch, 1 / config["solver"]["world_update_rate"])

    estimated_states = od.fit_orbit(starting_epoch, times, landmarks, pixel_coordinates, cubesat_attitudes)
    position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    rms_position_error = np.sqrt(np.sum(position_errors ** 2))
    print(f"RMS position error: {rms_position_error}")


if __name__ == "__main__":
    test_od()
