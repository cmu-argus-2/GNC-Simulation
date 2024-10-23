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
        b_dot_gain: float,
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
        
        self.ubm = 2.0*N_per_face * max_current * A_cross
        self.lbm = 0*self.ubm
        self.k = np.array(b_dot_gain)

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
        target_angular_velocity: np.ndarray,
        sun_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        h_tgt = self.J @ target_angular_velocity
        h_tgt_norm = np.linalg.norm(h_tgt)
        I_min_index = np.argmin(np.diag(self.J))
        I_min_direction = np.zeros(3)
        I_min_direction[I_min_index] = 1.0
        u = np.zeros(3)
        spin_stabilized = (np.linalg.norm(I_min_direction - (h/h_tgt_norm)) <= np.deg2rad(15))
        sun_pointing = (np.linalg.norm(sun_vector-h/h_norm)<= np.deg2rad(10))
        """
        detumbled  = (np.linalg.norm(angular_velocity) <= np.deg2rad(3))
        if not detumbled:
            magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
            unit_magnetic_field = magnetic_field / magnetic_field_norm
            m_cmd = -self.k @ np.cross(unit_magnetic_field, angular_velocity).reshape(3,1)
            m_cmd = m_cmd / magnetic_field_norm
            u = np.clip(a=m_cmd, a_min=self.lbm, a_max=self.ubm)
            print("Detumbling: Angular momentum h =", h)
        """
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)
        if not spin_stabilized:
            u = crossproduct(unit_magnetic_field) @ (I_min_direction - (h/h_tgt_norm))
            # u = np.clip(a=u, a_min=self.lbm, a_max=self.ubm)
            # u = self.k  @ np.cross(unit_magnetic_field, target_angular_velocity - angular_velocity).reshape(3,1)
            print(f"Spin-stabilizing: h = {h}, Norm of angular momentum h_norm = {h_norm}")
            print("h_tgt=", h_tgt)
            
        elif not sun_pointing:
            u = crossproduct(unit_magnetic_field) @ (sun_vector - (h/h_norm))
            print("Sun pointing: Sun vector =", sun_vector)
            print("Angular momentum direction =", h / h_norm)
        
        u = self.ubm * u/np.linalg.norm(u) 
        
        angle_sun_h = np.arccos(np.clip(np.dot(sun_vector, h / h_norm), -1.0, 1.0))
        print("Angle between sun vector and angular momentum direction (degrees):", np.degrees(angle_sun_h))
        print("torque command =", np.cross(u.T, magnetic_field))
        return u.reshape(3, 1)


class BaselineSunPointingController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        mtb_config: dict,
        b_dot_gain: float,
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
        self.lbm = -self.ubm
        self.k = np.array(b_dot_gain)

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
        target_angular_velocity: np.ndarray,
        sun_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        h_tgt = self.J @ target_angular_velocity
        h_tgt_norm = np.linalg.norm(h_tgt)
        I_max_index = np.argmax(np.diag(self.J))
        I_max_direction = np.zeros(3)
        I_max_direction[I_max_index] = 1.0
        u = np.zeros(3)
        spin_stabilized = (np.linalg.norm(I_max_direction - (h/h_tgt_norm)) <= np.deg2rad(15))
        sun_pointing = (np.linalg.norm(sun_vector-h/h_norm)<= np.deg2rad(10))

        α = 0.5
        u = crossproduct(magnetic_field) @ ((1-α)*(h_tgt_norm-h) + α*(sun_vector*h_norm-h))
        u = np.clip(a=u, a_min=self.lbm, a_max=self.ubm)
        print("torque: ", crossproduct(u) @ magnetic_field)
        return u.reshape(3, 1)

class BcrossController():
    def __init__(
        self,
        b_dot_gain: float,
        dipole_moment_lower_bound=-np.finfo(np.float64).max,
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
        unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)
        m_cmd = -self.k @ np.cross(unit_magnetic_field, angular_velocity).reshape(3,1)
        return np.clip(a=m_cmd, a_min=self.lbm, a_max=self.ubm)
