from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from brahe.epoch import Epoch

from world.physics.models.gravity import Gravity
from world.math.integrators import RK4
from world.math.quaternions import q_inv
from world.math.time import increment_epoch
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
        self.gravity = Gravity()

    def state_derivative(self, x: np.ndarray, u: None, epoch: Epoch) -> np.ndarray:
        """
        The state derivative function xdot = f(x, u).

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param u: The current control input (not used, but needed for RK4).
        :param epoch: The current epoch as an instance of brahe's Epoch class.
        :return: A numpy array of shape (6,) containing the state derivative.
        """
        r = x[:3]
        v = x[3:]
        a = self.gravity.acceleration(r, epc=epoch)
        return np.concatenate([v, a])

    def f(self, x: np.ndarray, epoch: Epoch) -> np.ndarray:
        """
        The state transition function x_{t+1} = f(x_t).

        :param x: A numpy array of shape (6,) containing the current state (position and velocity).
        :param epoch: The current epoch as an instance of brahe's Epoch class.
        :return: A numpy array of shape (6,) containing the next state (position and velocity).
        """
        return RK4(x, None, partial(self.state_derivative, epoch=epoch), self.dt)

    def fit_orbit(self, starting_epoch: Epoch, times: np.ndarray, landmarks: np.ndarray, pixel_coordinates: np.ndarray,
                  cubesat_attitudes: np.ndarray) -> np.ndarray:
        """
        Solve the orbit determination problem using non-linear least squares.

        :param starting_epoch: The starting epoch as an instance of brahe's Epoch class.
        :param times: A numpy array of shape (n,) and dtype of int containing the indices of time steps at which
                      landmarks were observed. Must be sorted in non-strictly ascending order.
        :param landmarks: A numpy array of shape (n, 3) containing the ECI coordinates of the landmarks.
        :param pixel_coordinates: A numpy array of shape (n, 2) containing the pixel coordinates of the landmarks.
        :param cubesat_attitudes: A numpy array of shape (n, 4) containing the quaternions representing the attitude of the satellite.
        :return: A numpy array of shape (max(times) + 1, 6) containing the ECI position and velocity of the satellite at each time step.
        """
        assert len(times.shape) == 1, "times must be a 1D array"
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), \
            "times must be sorted in non-strictly ascending order"
        assert len(landmarks.shape) == 2, "landmarks must be a 2D array"
        assert landmarks.shape[1] == 3, "landmarks must have 3 columns"
        assert len(pixel_coordinates.shape) == 2, "pixel_coordinates must be a 2D array"
        assert pixel_coordinates.shape[1] == 2, "pixel_coordinates must have 2 columns"
        assert len(times) == len(landmarks) == len(pixel_coordinates), \
            "times, landmarks, and pixel_coordinates must have the same length"

        bearing_vectors = self.camera.convert_pixel_coordinates_to_camera_ray_directions(pixel_coordinates)
        bearing_unit_vectors = np.linalg.norm(bearing_vectors, axis=1)

        eci_to_camera_rotations = np.zeros((len(times), 3, 3))
        for i in range(len(times)):
            eci_to_camera_rotations[i, ...] = self.camera.R_sat_cam @ Rotation.from_quat(q_inv(cubesat_attitudes[i, :]), scalar_first=True).as_matrix()

        N = times[-1] + 1  # number of time steps

        def residuals(X: np.ndarray) -> np.ndarray:
            """
            Compute the residuals of the non-linear least squares problem.

            :param X: A flattened numpy array of shape (6 * N,) containing the ECI positions and velocities of the satellite at each time step.
            :return: A numpy array of shape (6 * (N - 1) + 3 * len(times),) containing the residuals.
            """
            states = X.reshape(-1, 6)
            res = np.zeros(6 * (N - 1) + 3 * len(times))

            # dynamics residuals
            epoch = starting_epoch
            for i in range(N - 1):
                res[6 * i:6 * (i + 1)] = states[i + 1, :] - self.f(states[i, :], epoch)
                epoch = increment_epoch(epoch, self.dt)

            # measurement residuals
            for i, (time, landmark, eci_to_camera_rotation) in enumerate(zip(times, landmarks, eci_to_camera_rotations)):
                cubesat_position = states[time, :3]
                predicted_bearing = eci_to_camera_rotation @ (landmark - cubesat_position)  # in camera frame
                res[6 * (N - 1) + 3 * i:6 * (N - 1) + 3 * (i + 1)] = bearing_unit_vectors[i] - predicted_bearing / np.linalg.norm(predicted_bearing)

            return res

        initial_guess = np.ones(6 * N)  # TODO: improve initial guess
        result = least_squares(residuals, initial_guess, method="lm")
        return result.x.reshape(-1, 6)
