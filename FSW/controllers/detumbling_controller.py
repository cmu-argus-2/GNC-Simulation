import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *


Z = np.block([
    [np.eye(3),       np.zeros((3,3))],
    [np.zeros((3,3)), np.zeros((3,3))],
])


def convert_to_skew_symmetric(
        vec: np.ndarray,
    ) -> np.ndarray:
        return np.array([
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ])


class NonMonotonicController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        time_step: float,
        alpha=100.0,
        beta=1.0,
        dipole_moment_lower_bound=np.zeros(3),
        dipole_moment_upper_bound=np.finfo(np.float64).max,
    ) -> None:
        self.J = inertia_tensor
        self.dt = time_step
        self.alpha = alpha
        self.beta = beta
        self.lbm = dipole_moment_lower_bound
        self.ubm = dipole_moment_upper_bound

    def get_dipole_moment_command(
        self,
        magnetic_field_in_eci_frame: np.ndarray,
        magnetic_field_rate_in_body_frame: np.ndarray,
        angular_velocity_in_eci_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Discrete non-monotonic law:
            https://rexlab.ri.cmu.edu/papers/nonmonotonic_detumbling_ieee24.pdf
        """
        Bdot_eci = convert_to_skew_symmetric(angular_velocity_in_eci_frame) \
                   * magnetic_field_in_eci_frame \
                   + magnetic_field_rate_in_body_frame
        B1 = magnetic_field_in_eci_frame + self.dt * Bdot_eci
        B1_hat = convert_to_skew_symmetric(B1)
        B0_hat = convert_to_skew_symmetric(magnetic_field_in_eci_frame)
        B_bar = np.vstack((B0_hat, B1_hat))

        Q1 = self.dt**2 * Z @ B_bar @ B_bar.T @ Z
        Q2 = self.alpha * self.dt**2 * B_bar @ B_bar.T

        h0 = self.J @ angular_velocity_in_eci_frame
        q1 = self.dt * (h0.T @ B_bar.T @ Z).T
        q2 = self.alpha * self.dt * (h0.T @ B_bar.T).T

        mu_bar = np.linalg.inv( self.beta * np.eye(6) + Q1 + self.alpha * Q2 ) \
                    @ ( q1 + self.alpha * q2 )
        return np.clip(a=mu_bar[:2], a_min=self.lbm, a_max=self.ubm)


class BcrossController():
    def __init__(
        self,
        b_dot_gain: float,
        dipole_moment_lower_bound=np.zeros(3),
        dipole_moment_upper_bound=np.finfo(np.float64).max,
    ) -> None:
        self.k = b_dot_gain
        self.lbm = dipole_moment_lower_bound
        self.ubm = dipole_moment_upper_bound

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        B-cross law: https://arc.aiaa.org/doi/epdf/10.2514/1.53074
        """
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        unit_magnetic_field = magnetic_field / magnetic_field_norm
        m_cmd = -self.k / magnetic_field_norm \
                        * np.cross(unit_magnetic_field, angular_velocity)
        return np.clip(a=m_cmd, a_min=self.lbm, a_max=self.ubm)
