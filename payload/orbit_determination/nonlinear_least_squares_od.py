from collections.abc import Callable
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.stats import circmean, circvar

from world.math.quaternions import q_inv
from world.physics.orbital_dynamics import f, f_jac
from sensors.mock_vision_model import Camera

from brahe.constants import R_EARTH, GM_EARTH


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

    @staticmethod
    def fit_circular_orbit(times: np.ndarray, positions: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """
        Fits a circular orbit model to a set of timestamped ECI position estimates.
        This is used for creating an initial guess for the non-linear least squares problem.

        :param times: The timestamps as a numpy array of shape (n,).
        :param positions: The position estimates in ECI as a numpy array of shape (n, 3).
        :return: A function that maps timestamps to position estimates based on the resulting circular orbit model.
        """
        assert len(positions.shape) == 2, "positions must be a 2D array"
        assert positions.shape[1] == 3, "positions must have 3 columns"
        assert positions.shape[0] >= 3, "There must be at least 3 points"
        assert len(times.shape) == 1, "times must be a 1D array"
        assert len(times) == positions.shape[0], "times and positions must have the same length"

        """
        We want to solve for the unit normal vector of the best fit plane that passes through the origin. 
        This means we want to minimize np.linalg.norm(positions @ normal) subject to np.linalg.norm(normal) == 1.
        This is solved by computing the right singular vector corresponding to the smallest singular value of positions.
        """
        *_, Vt = np.linalg.svd(positions)
        normal = Vt[-1, :]

        projected_positions = positions - np.outer(positions @ normal, normal)
        orbital_radius = np.mean(np.linalg.norm(projected_positions, axis=1))
        speed = np.sqrt(GM_EARTH / orbital_radius)
        angular_speed = speed / orbital_radius

        # choose basis vectors for the orbital plane
        # there are 2 cases to avoid scenarios where the cross product is close to zero
        if abs(normal[0]) <= abs(normal[1]):
            # normal is closer to the y-axis, so we will use the x-axis in the cross product
            y_axis = np.cross(normal, np.array([1, 0, 0]))
            y_axis = y_axis / np.linalg.norm(y_axis)
            if y_axis[1] < 0:
                y_axis = -y_axis
            x_axis = np.cross(y_axis, normal)
        else:
            # normal is closer to the x-axis, so we will use the y-axis in the cross product
            x_axis = np.cross(np.array([0, 1, 0]), normal)
            x_axis = x_axis / np.linalg.norm(x_axis)
            if x_axis[0] < 0:
                x_axis = -x_axis
            y_axis = np.cross(normal, x_axis)

        positions_2d = projected_positions @ np.column_stack((x_axis, y_axis))
        angles = np.arctan2(positions_2d[:, 1], positions_2d[:, 0])

        # case 1: the orbit is counter-clockwise in the chosen basis
        phases_ccw = angles - angular_speed * times

        # case 2: the orbit is clockwise in the chosen basis
        phases_cw = angles + angular_speed * times

        # choose the case with the smaller variance
        if circvar(phases_ccw) < circvar(phases_cw):
            angular_velocity = angular_speed
            angle_0 = circmean(phases_ccw)
        else:
            angular_velocity = -angular_speed
            angle_0 = circmean(phases_cw)

        def model(ts: np.ndarray) -> np.ndarray:
            """
            Maps timestamps to state estimates.

            :param ts: The timestamps to map to state estimates as a numpy array of shape (m,).
            :return: The resulting state estimates as a numpy array of shape (m, 6).
            """
            angles_ = angular_velocity * ts + angle_0
            positions_2d_ = orbital_radius * np.column_stack((np.cos(angles_), np.sin(angles_)))
            positions_ = positions_2d_ @ np.row_stack((x_axis, y_axis))

            velocity_directions_ = np.sign(angular_speed) * np.cross(normal, positions_)
            velocity_directions_ = velocity_directions_ / np.linalg.norm(velocity_directions_, axis=1, keepdims=True)
            velocities_ = speed * velocity_directions_

            return np.column_stack((positions_, velocities_))

        return model

    def fit_orbit(self, times: np.ndarray, landmarks: np.ndarray, pixel_coordinates: np.ndarray,
                  cubesat_attitudes: np.ndarray, N: int = None,
                  semi_major_axis_guess: float = R_EARTH + 600e3) -> np.ndarray:
        """
        Solve the orbit determination problem using non-linear least squares.

        :param times: A numpy array of shape (m,) and dtype of int containing the indices of time steps at which
                      landmarks were observed. Must be sorted in non-strictly ascending order.
        :param landmarks: A numpy array of shape (m, 3) containing the ECI coordinates of the landmarks.
        :param pixel_coordinates: A numpy array of shape (m, 2) containing the pixel coordinates of the landmarks.
        :param cubesat_attitudes: A numpy array of shape (m, 4) containing the quaternions representing the attitude of the satellite.
        :param N: The number of time steps. If None, it will be set to the maximum value in times plus one.
        :param semi_major_axis_guess: An initial guess for the semi-major axis of the satellite's orbit.
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

        altitude_normalized_landmarks = landmarks / np.linalg.norm(landmarks, axis=1, keepdims=True)
        model = OrbitDetermination.fit_circular_orbit(times, semi_major_axis_guess * altitude_normalized_landmarks)
        initial_guess = model(np.arange(N) * self.dt).flatten()

        result = least_squares(residuals, initial_guess, method="lm", jac=residual_jac)
        return result.x.reshape(N, 6)
