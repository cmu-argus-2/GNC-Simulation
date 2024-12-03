from abc import ABC, abstractmethod
from typing import Any, Tuple
from time import perf_counter
from datetime import datetime
import yaml
import os

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from matplotlib import use

import brahe
from brahe.epoch import Epoch
from brahe.constants import R_EARTH, GM_EARTH

from world.math.time import increment_epoch
from world.math.transforms import update_brahe_data_files
from world.physics.orbital_dynamics import f
from sensors.mock_vision_model2 import Camera
from payload.image_simulation.earth_vis import EarthImageSimulator, lat_lon_to_ecef
from payload.vision.ml_pipeline import MLPipeline
from payload.vision.camera import Frame
from nonlinear_least_squares_od import OrbitDetermination


"""
TODO:
- Perform camera calibration
- Project landsat data to simulate images
- Play with the vision model
- Implement outlier rejection
Need from Ibra
- How to get images off of the cameras
- How to run the vision model
- Walkthrough of Kyle's landsat projection code
"""


class LandmarkBearingSensor(ABC):
    """
    Abstract class for a landmark bearing sensor, which inputs the satellite pose and outputs landmark bearing measurements.
    """

    @abstractmethod
    def take_measurement(self, epoch: Epoch, cubesat_position: np.ndarray, R_body_to_eci: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a landmark bearing measurement using the sensor.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        pass


class RandomLandmarkBearingSensor(LandmarkBearingSensor):
    """
    A sensor that randomly generates landmark bearing measurements within a cone centered about the camera's boresight.
    """

    def __init__(self, config, max_measurements: int = 10):
        """
        :param config: The configuration dictionary.
        :param max_measurements: The number of measurements to attempt to take at once. The actual number may be less.
        """
        camera_params = config["satellite"]["camera"]
        R_body_to_camera = Rotation.from_quat(np.asarray(camera_params["orientation_in_cubesat_frame"]),
                                              scalar_first=True).as_matrix()
        self.camera = Camera(
            fov=np.deg2rad(120),
            R_body_to_camera=R_body_to_camera,
            t_body_to_camera=np.asarray(camera_params["position_in_cubesat_frame"])
        )
        self.max_measurements = max_measurements

    def take_measurement(self, _: Epoch, cubesat_position: np.ndarray, R_body_to_eci: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.
        The number of measurements, N, will be some number less than or equal to self.max_measurements.

        :param _: The epoch as an instance of brahe's Epoch class. Not used.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        bearing_unit_vectors = self.camera.sample_bearing_unit_vectors(self.max_measurements)
        valid_intersections, landmark_positions_eci = self.camera.get_earth_intersections(bearing_unit_vectors,
                                                                                          cubesat_position,
                                                                                          R_body_to_eci)
        bearing_unit_vectors = bearing_unit_vectors[valid_intersections, :]

         # sanity check
        for bearing_unit_vector, landmark_position_eci in zip(bearing_unit_vectors, landmark_positions_eci):
            true_bearing_unit_vector = landmark_position_eci - cubesat_position
            true_bearing_unit_vector /= np.linalg.norm(true_bearing_unit_vector)
            assert np.allclose(true_bearing_unit_vector, R_body_to_eci @ bearing_unit_vector)

        return bearing_unit_vectors, landmark_positions_eci


class SimulatedMLLandmarkBearingSensor:
    """
    A sensor that simulates an image of the Earth from the camera's pose and runs the ML pipeline to generate landmark bearing measurements.
    """

    def __init__(self, config):
        """
        :param config: The configuration dictionary.
        """
        quat_body_to_camera = np.asarray(config["satellite"]["camera"]["orientation_in_cubesat_frame"])
        self.R_camera_to_body = Rotation.from_quat(quat_body_to_camera, scalar_first=True).inv().as_matrix()
        self.ml_pipeline = MLPipeline()
        self.earth_image_simulator = EarthImageSimulator()

    def take_measurement(self, epoch: Epoch, cubesat_position: np.ndarray, R_body_to_eci: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Take a set of landmark bearing measurements.

        :param epoch: The epoch as an instance of brahe's Epoch class.
        :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
        :param R_body_to_eci: The rotation matrix from the body frame to the ECI frame as a numpy array of shape (3, 3).
        :return: A tuple containing a numpy array of shape (N, 3) containing the bearing unit vectors in the body frame
                 and a numpy array of shape (N, 3) containing the landmark positions in ECI coordinates.
        """
        R_eci_to_ecef = brahe.frames.rECItoECEF(epoch)
        position_ecef = R_eci_to_ecef @ cubesat_position
        R_body_to_ecef = R_eci_to_ecef @ R_body_to_eci

        print(f"Taking measurement at {epoch=}, {cubesat_position=}, {R_body_to_eci=}")

        # simulate image
        image = self.earth_image_simulator.simulate_image(position_ecef, R_body_to_ecef)

        if np.all(image == 0):
            print("No image detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        # run the ML pipeline on the image
        frame = Frame(image, 0, datetime.now())
        # TODO: queue requests to the model and send them in batches as the sim runs
        regions_and_landmarks = self.ml_pipeline.run_ml_pipeline_on_single(frame)

        # save the image with the detected landmarks
        epoch_str = str(epoch) \
            .replace(':', '_') \
            .replace(' ', '_') \
            .replace('.', '_')
        output_dir = os.path.abspath(
            os.path.join(__file__, f"../log/simulated_images/seed_69420_epoch_{epoch_str}/"))
        self.ml_pipeline.visualize_landmarks(frame, regions_and_landmarks, output_dir)

        landmark_positions_ecef = np.zeros(shape=(0, 3))
        pixel_coordinates = np.zeros(shape=(0, 2))
        confidence_scores = np.zeros(shape=(0,))

        for region, landmarks in regions_and_landmarks:
            centroids_ecef = lat_lon_to_ecef(landmarks.centroid_latlons[np.newaxis, ...]).reshape(-1, 3)

            landmark_positions_ecef = np.concatenate((landmark_positions_ecef, centroids_ecef), axis=0)
            pixel_coordinates = np.concatenate((pixel_coordinates, landmarks.centroid_xy), axis=0)
            confidence_scores = np.concatenate((confidence_scores, landmarks.confidence_scores), axis=0)

        if len(confidence_scores) == 0:
            print("No landmarks detected")
            return np.zeros(shape=(0, 3)), np.zeros(shape=(0, 3))

        landmark_positions_eci = (R_eci_to_ecef.T @ landmark_positions_ecef.T).T
        bearing_unit_vectors_cf = self.earth_image_simulator.camera.pixel_to_bearing_unit_vector(pixel_coordinates)
        bearing_unit_vectors = (self.R_camera_to_body @ bearing_unit_vectors_cf.T).T

        print(f"Detected {len(landmark_positions_eci)} landmarks")

        # TODO: output confidence_scores too
        return bearing_unit_vectors, landmark_positions_eci


def load_config() -> dict[str, Any]:
    """
    Load the configuration file and modify it for the purposes of this test.

    :return: The modified configuration file as a dictionary.
    """
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # decrease world update rate since we only care about position dynamics
    config["solver"]["world_update_rate"] = 1 / 60  # Hz
    config["mission"]["duration"] = 3 * 90 * 60  # s, roughly 1 orbit

    return config


def get_nadir_rotation(cubesat_position: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix from the body frame to the ECI frame for a satellite with an orbital angular momentum in the -y direction.

    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: A numpy array of shape (3, 3) containing the rotation matrix from the body frame to the ECI frame.
    """
    y_axis = [0, -1, 0]  # along orbital angular momentum
    z_axis = -cubesat_position / np.linalg.norm(cubesat_position)  # along radial vector
    x_axis = np.cross(y_axis, z_axis)
    R_body_to_eci = np.column_stack([x_axis, y_axis, z_axis])
    return R_body_to_eci


def is_over_daytime(epoch: Epoch, cubesat_position: np.ndarray) -> bool:
    """
    Determine if the satellite is above a portion of the Earth that is in daylight.

    :param epoch: The epoch as an instance of brahe's Epoch class.
    :param cubesat_position: The position of the satellite in ECI as a numpy array of shape (3,).
    :return: True if the satellite is above the daylight portion of the Earth, False otherwise.
    """
    return np.dot(brahe.ephemerides.sun_position(epoch), cubesat_position) > 0


def get_SO3_noise_matrices(N: int, magnitude_std: float) -> np.ndarray:
    """
    Generate a set of matrices representing random rotations in SO(3) with a given standard deviation.

    :param N: The number of noise matrices to generate.
    :param magnitude_std: The standard deviation of the magnitudes of the rotations in radians.
    :return: A numpy array of shape (N, 3, 3) containing the noise rotations.
    """
    magnitudes = np.abs(np.random.normal(scale=magnitude_std, size=N))
    directions = np.random.normal(size=(N, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return Rotation.from_rotvec(magnitudes[:, np.newaxis] * directions).as_matrix()


def animate_orbits(positions: np.ndarray, estimated_positions: np.ndarray, landmarks: np.ndarray) -> None:
    """
    Creates an animation where the orbital paths of the true and estimated states are plotted as evolving over time.
    The landmarks are also plotted statically.

    :param positions: A numpy array of shape (N, 3) containing the true positions.
    :param estimated_positions: A numpy array of shape (N, 3) containing the estimated positions.
    :param landmarks: A numpy array of shape (M, 3) containing the landmark positions.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for t in range(positions.shape[0]):
        start_idx = max(0, t - 100)
        ax.clear()
        ax.plot(positions[start_idx:t, 0],
                positions[start_idx:t, 1],
                positions[start_idx:t, 2],
                label="True orbit")
        ax.plot(estimated_positions[start_idx:t, 0],
                estimated_positions[start_idx:t, 1],
                estimated_positions[start_idx:t, 2],
                label="Estimated orbit")
        ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], label="Landmarks")

        ax.set_xlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_ylim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_zlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.pause(0.01)


def test_od():
    # update_brahe_data_files()
    config = load_config()

    # set up landmark bearing sensor and orbit determination objects
    landmark_bearing_sensor = RandomLandmarkBearingSensor(config)
    # landmark_bearing_sensor = SimulatedMLLandmarkBearingSensor(config)
    od = OrbitDetermination(dt=1 / config["solver"]["world_update_rate"])

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
        Reads from the states and landmark_bearing_sensor variables in the outer scope.
        Appends to the times, Rs_body_to_eci, bearing_unit_vectors, and landmarks arrays in the outer scope.

        :param t_idx: The time index at which to take the measurements.
        """
        position = states[t_idx, :3]
        R_body_to_eci = get_nadir_rotation(position)

        measurement_bearing_unit_vectors, measurement_landmarks = landmark_bearing_sensor.take_measurement(epoch, position, R_body_to_eci)
        measurement_count = measurement_bearing_unit_vectors.shape[0]
        assert measurement_landmarks.shape[0] == measurement_count

        nonlocal times, Rs_body_to_eci, bearing_unit_vectors, landmarks
        times = np.concatenate((times, np.repeat(t_idx, measurement_count)))
        Rs_body_to_eci = np.concatenate((Rs_body_to_eci,
                                         np.tile(R_body_to_eci, (measurement_count, 1, 1))),
                                        axis=0)
        bearing_unit_vectors = np.concatenate((bearing_unit_vectors, measurement_bearing_unit_vectors), axis=0)
        landmarks = np.concatenate((landmarks, measurement_landmarks), axis=0)
        print(f"Total measurements so far: {len(times)}")
        print(f"Completion: {100 * t_idx / N:.2f}%")

    for t in range(0, N - 1):
        states[t + 1, :] = f(states[t, :], od.dt)

        if t % 5 == 0 and is_over_daytime(epoch, states[t, :3]):  # take a set of measurements every 5 minutes
            take_measurement(t)

        epoch = increment_epoch(epoch, 1 / config["solver"]["world_update_rate"])

    if len(times) == 0:
        raise ValueError("No measurements taken")

    # so3_noise_matrices = get_SO3_noise_matrices(len(times), np.deg2rad(0.001))
    # bearing_unit_vectors = np.einsum("ijk,ik->ij", so3_noise_matrices, bearing_unit_vectors)

    start_time = perf_counter()
    estimated_states = od.fit_orbit(times, landmarks, bearing_unit_vectors, Rs_body_to_eci, N)
    print(f"Elapsed time: {perf_counter() - start_time:.2f} s")

    position_errors = np.linalg.norm(states[:, :3] - estimated_states[:, :3], axis=1)
    rms_position_error = np.sqrt(np.mean(position_errors ** 2))
    print(f"RMS position error: {rms_position_error}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    ax.set_ylim(-1.5 * R_EARTH, 1.5 * R_EARTH)
    ax.set_zlim(-1.5 * R_EARTH, 1.5 * R_EARTH)

    ax.plot(states[:, 0], states[:, 1], states[:, 2], label="True orbit")
    ax.plot(estimated_states[:, 0], estimated_states[:, 1], estimated_states[:, 2], label="Estimated orbit")
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], label="Landmarks")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()


def fake_plots():
    plt.figure()
    plt.xlabel("Attitude SO(3) Noise Variance [deg]")
    plt.ylabel("OD RMS Position Error [km]")
    plt.title("Effect of Attitude Noise on Orbit Determination")
    xs = np.linspace(0, 10, 50)
    ys = np.abs(3 * xs + 5 + np.random.normal(scale=3, size=xs.shape))
    plt.scatter(xs, ys)
    plt.plot([0, 10], [50, 50], linestyle="--", color="r")
    plt.show()

    plt.figure()
    plt.xlabel("OD RMS Position Error [km]")
    plt.ylabel("Frequency")
    plt.title("Histogram of RMS Position Error")
    ys = np.sort(np.random.normal(loc=35, scale=5, size=100))
    plt.hist(ys, bins=10)
    plt.plot([50, 50], [0, 20], linestyle="--", color="r")
    plt.show()


if __name__ == "__main__":
    np.random.seed(69420)
    use("TkAgg")
    test_od()
