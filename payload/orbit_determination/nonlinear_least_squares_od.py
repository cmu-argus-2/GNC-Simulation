import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from world.math.quaternions import q_inv
from world.physics.orbital_dynamics import f, f_jac
from sensors.mock_vision_model import Camera


class OrbitDetermination:
    """
    A class for solving the orbit determination problem using non-linear least squares.
    Note that this class uses its own dynamics model since it needs to be decoupled from the attitude dynamics.
    """

    def __init__(self, camera: Camera, dt: float) -> None:
        """
        Initialize the OrbitDetermination object.

        :param camera: An instance of the Camera class.
        :param dt: The amount of time between each time step.
        """
        self.camera = camera
        self.dt = dt

    def fit_orbit(self, times: np.ndarray, landmarks: np.ndarray, pixel_coordinates: np.ndarray,
                  cubesat_attitudes: np.ndarray, N: int = None) -> np.ndarray:
        """
        Solve the orbit determination problem using non-linear least squares.

        :param times: A numpy array of shape (n,) and dtype of int containing the indices of time steps at which
                      landmarks were observed. Must be sorted in non-strictly ascending order.
        :param landmarks: A numpy array of shape (m, 3) containing the ECI coordinates of the landmarks.
        :param pixel_coordinates: A numpy array of shape (m, 2) containing the pixel coordinates of the landmarks.
        :param cubesat_attitudes: A numpy array of shape (m, 4) containing the quaternions representing the attitude of the satellite.
        :param N: The number of time steps. If None, it will be set to the maximum value in times plus one.
        :return: A numpy array of shape (N, 6) containing the ECI position and velocity of the satellite at each time step.
        """
        assert len(times.shape) == 1, "times must be a 1D array"
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), \
            "times must be sorted in non-strictly ascending order"
        assert len(landmarks.shape) == 2, "landmarks must be a 2D array"
        assert landmarks.shape[1] == 3, "landmarks must have 3 columns"
        assert len(pixel_coordinates.shape) == 2, "pixel_coordinates must be a 2D array"
        assert pixel_coordinates.shape[1] == 2, "pixel_coordinates must have 2 columns"
        assert len(cubesat_attitudes.shape) == 2, "cubesat_attitudes must be a 2D array"
        assert cubesat_attitudes.shape[1] == 4, "cubesat_attitudes must have 4 columns"
        assert len(times) == len(landmarks) == len(pixel_coordinates) == len(cubesat_attitudes), \
            "times, landmarks, pixel_coordinates, and cubesat_attitudes must have the same length"
        if N is None:
            N = times[-1] + 1  # number of time steps
        else:
            assert N > times[-1], "N must be greater than the maximum value in times"

        bearing_vectors = self.camera.convert_pixel_coordinates_to_camera_ray_directions(pixel_coordinates)
        bearing_unit_vectors = bearing_vectors / np.linalg.norm(bearing_vectors, axis=1, keepdims=True)

        eci_to_camera_rotations = np.zeros((len(times), 3, 3))
        for i in range(len(times)):
            eci_to_camera_rotations[i, ...] = self.camera.R_sat_cam @ Rotation.from_quat(q_inv(cubesat_attitudes[i, :]), scalar_first=True).as_matrix()

        def residuals(X: np.ndarray) -> np.ndarray:
            """
            Compute the residuals of the non-linear least squares problem.

            :param X: A flattened numpy array of shape (6 * N,) containing the ECI positions and velocities of the satellite at each time step.
            :return: A numpy array of shape (6 * (N - 1) + 3 * len(times),) containing the residuals.
            """
            states = X.reshape(N, 6)
            res = np.zeros(6 * (N - 1) + 3 * len(times))

            # dynamics residuals
            for i in range(N - 1):
                res[6 * i:6 * (i + 1)] = states[i + 1, :] - f(states[i, :], self.dt)

            # measurement residuals
            for i, (time, landmark, eci_to_camera_rotation) in enumerate(zip(times, landmarks, eci_to_camera_rotations)):
                cubesat_position = states[time, :3]
                predicted_bearing = eci_to_camera_rotation @ (landmark - cubesat_position)  # in camera frame
                res[6 * (N - 1) + 3 * i:6 * (N - 1) + 3 * (i + 1)] = bearing_unit_vectors[i] - predicted_bearing / np.linalg.norm(predicted_bearing)

            return res

        def residual_jac(X: np.ndarray):
            """
            Compute the Jacobian of the residuals of the non-linear least squares problem.

            :param X: A flattened numpy array of shape (6 * N,) containing the ECI positions and velocities of the satellite at each time step.
            :return: A numpy array of shape (6 * (N - 1) + 3 * len(times), 6 * N) containing the Jacobian of the residuals.
            """
            states = X.reshape(N, 6)
            jac = np.zeros((6 * (N - 1) + 3 * len(times), 6 * N), dtype=X.dtype)
            # Note that indices into the columns of jac are 6 * i : 6 * (i + 1) for the ith state

            # dynamics Jacobian
            for i in range(N - 1):
                jac[6 * i:6 * (i + 1), 6 * (i + 1):6 * (i + 2)] = np.eye(6)
                jac[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = -f_jac(states[i, :], self.dt)

            # measurement Jacobian
            for i, (time, landmark, eci_to_camera_rotation) in enumerate(
                    zip(times, landmarks, eci_to_camera_rotations)):
                cubesat_position = states[time, :3]
                predicted_bearing = eci_to_camera_rotation @ (landmark - cubesat_position)
                predicted_bearing_norm = np.linalg.norm(predicted_bearing)
                predicted_bearing_unit_vector = predicted_bearing / predicted_bearing_norm
                jac[6 * (N - 1) + 3 * i:6 * (N - 1) + 3 * (i + 1), 6 * time:6 * time + 3] = \
                    (np.outer(predicted_bearing_unit_vector, predicted_bearing_unit_vector) - np.eye(3)) @ eci_to_camera_rotation / predicted_bearing_norm

            return jac

        initial_guess = np.ones(6 * N)  # TODO: improve initial guess
        result = least_squares(residuals, initial_guess, method="lm", jac=residual_jac)
        return result.x.reshape(N, 6)
