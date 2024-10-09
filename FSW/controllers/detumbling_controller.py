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


class LyapBasedSunPointingController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        mtb_config: dict,
    ) -> None:
        # [TODO] max moment should be defined per axis based on all magnetorquers, not just first
        self.J = inertia_tensor        
        max_voltage = mtb_config["max_voltage"][0]
        max_power = mtb_config["max_power"][0]
        max_current_rating = mtb_config["max_current_rating"][0] 
        max_current = np.min([max_power / max_voltage, max_current_rating])
        trace_thickness = mtb_config["trace_thickness"][0] 
        N = mtb_config["coils_per_layer"][0]
        trace_width = mtb_config["trace_width"][0] 
        gap_width = mtb_config["gap_width"][0] 
        coil_width = trace_width + gap_width
        pcb_layers = mtb_config["layers"][0]
        N_per_face = N * pcb_layers
        pcb_side_max = 0.1
        A_cross = (pcb_side_max - N * coil_width) ** 2
        coil_length = 4 * (pcb_side_max - N*coil_width) \
                        * N * pcb_layers
        COPPER_RESISTIVITY = 1.724 * 10**-8

        R =  COPPER_RESISTIVITY * coil_length \
            / (trace_width * trace_thickness)
        
        self.ubm = N_per_face * max_current * A_cross

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
        sc_inertia: np.ndarray,
        target_angular_velocity: np.ndarray,
        sun_vector: np.ndarray,
        max_dipole_moment: float,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        h = sc_inertia @ angular_velocity 
        h_norm = np.linalg.norm(h)
        h_tgt = sc_inertia @ target_angular_velocity
        h_tgt_norm = np.linalg.norm(h_tgt)
        I_max_index = np.argmax(np.diag(sc_inertia))
        I_max_direction = np.zeros(3)
        I_max_direction[I_max_index] = 1.0
        if np.linalg.norm(I_max_direction - h / h_tgt_norm) > 0.1:
            u = magnetic_field @ (I_max_direction - h/h_tgt_norm)
            u = max_dipole_moment * u/np.linalg.norm(u)
        elif np.linalg.norm(sun_vector-h/h_norm)>0.01:
            u = magnetic_field @ (sun_vector - h/h_norm)
            u = max_dipole_moment * u/np.linalg.norm(u)
        else:
            u = np.zeros(3)
        
        return u


class BcrossController():
    def __init__(
        self,
        b_dot_gain: float,
        dipole_moment_lower_bound=np.zeros(3),
        dipole_moment_upper_bound=np.finfo(np.float64).max,
    ) -> None:
        self.k = np.array(b_dot_gain)
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
                        @ np.cross(unit_magnetic_field, angular_velocity).reshape(3,1)
        return np.clip(a=m_cmd, a_min=self.lbm.reshape(3,1), a_max=self.ubm)
