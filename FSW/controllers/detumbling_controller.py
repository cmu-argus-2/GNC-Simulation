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
        G_mtb: np.ndarray,
        b_dot_gain: float,
        target_angular_velocity: float,
        Magnetorquers: list,
    ) -> None:
        max_moms = np.zeros(len(Magnetorquers))
        for i, mtb in enumerate(Magnetorquers):
            max_moms[i] = mtb.max_dipole_moment
        
        self.ubm = np.min(np.abs(G_mtb).T @ max_moms)
        self.lbm = -self.ubm
        self.k = np.array(b_dot_gain)
        
        self.J = inertia_tensor 
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction 
        self.h_tgt = self.J @ self.I_min_direction * np.deg2rad(target_angular_velocity)
        self.h_tgt_norm = np.linalg.norm(self.h_tgt)

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
        u = np.zeros(3)
        spin_stabilized = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))
        sun_pointing = (np.linalg.norm(sun_vector-(h/h_norm))<= np.deg2rad(10))
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
        unit_magnetic_field = magnetic_field # / (magnetic_field_norm ** 2)
        if not spin_stabilized:
            u = crossproduct(unit_magnetic_field) @ (self.I_min_direction - (h/self.h_tgt_norm))
            # u = np.clip(a=u, a_min=self.lbm, a_max=self.ubm)
            # u = self.k  @ np.cross(unit_magnetic_field, target_angular_velocity - angular_velocity).reshape(3,1)
            # print(f"Spin-stabilizing: h = {h}, Norm of angular momentum h_norm = {h_norm}")
            # print("h_tgt=", self.h_tgt)
            
        elif not sun_pointing:
            u = crossproduct(unit_magnetic_field) @ (sun_vector - (h/self.h_tgt_norm)) # (h/self.h_tgt_norm))
            # print("Sun pointing: Sun vector =", sun_vector)
            # print("Angular momentum direction =", h / h_norm)
        
        if np.linalg.norm(u) == 0:
            u = np.zeros(3)
        else:
            u = self.ubm * u/np.linalg.norm(u) 
        
        angle_sun_h = np.arccos(np.clip(np.dot(sun_vector, h / h_norm), -1.0, 1.0))
        # print("Sun Vector: ", sun_vector)
        # print("Angle between sun vector and angular momentum direction (degrees):", np.degrees(angle_sun_h))
        # print("Angle between sun vector and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(sun_vector, self.I_min_direction))))
        # print("Angle between angular momentum and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(h/h_norm, self.I_min_direction))))
        # print("torque command =", np.cross(u.T, magnetic_field))
        return u.reshape(3, 1)


class BaselineSunPointingController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        G_mtb: np.ndarray,
        b_dot_gain: float,
        pd_gain: np.ndarray,
        target_angular_velocity: np.ndarray,
        Magnetorquers: list,
    ) -> None:
        # [TODO] max moment should be defined per axis based on all magnetorquers, not just first
        max_moms = np.zeros(len(Magnetorquers))
        for i, mtb in enumerate(Magnetorquers):
            max_moms[i] = mtb.max_dipole_moment
        
        self.ubm = np.min(np.abs(G_mtb).T @ max_moms)
        self.lbm = -self.ubm
        self.kdetumb = np.array(b_dot_gain)
        self.k = np.array(pd_gain)
        
        self.J = inertia_tensor 
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction 
        self.target_angular_velocity = target_angular_velocity
        self.ref_angular_velocity = self.I_min_direction * np.deg2rad(target_angular_velocity)
        self.h_tgt = self.J @ self.ref_angular_velocity
        self.h_tgt_norm = np.linalg.norm(self.h_tgt)

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
        
        u = np.zeros(3)
        spin_stabilized = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))
        sun_pointing = (np.linalg.norm(sun_vector-(h/h_norm))<= np.deg2rad(10))
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        """
        unit_magnetic_field = magnetic_field / (magnetic_field_norm)
        α = 0.5
        u = crossproduct(unit_magnetic_field) @ ((1-α)*((self.h_tgt-h) / self.h_tgt_norm) 
                                                 + α*((sun_vector*self.h_tgt_norm-h) / h_norm))
        """
        unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)

        if not spin_stabilized:
            u = self.kdetumb * np.cross(unit_magnetic_field, self.ref_angular_velocity - angular_velocity).reshape(3,1)
        elif not sun_pointing:
            Bhat = crossproduct(magnetic_field)
            U, S, V = np.linalg.svd(Bhat)
            idx = np.where(S > 1e-6 * np.max(S))[0]
            S_inv = np.diag([1/s if s > 1e-10 else 0 for s in S])
            Bhat_pseudo_inv = V.T @ S_inv @ U.T

            err_vec = np.hstack((-self.I_min_direction-sun_vector, self.ref_angular_velocity - angular_velocity))
            u = Bhat_pseudo_inv @ self.k @ err_vec

        u = np.clip(a=u, a_min=self.lbm, a_max=self.ubm)
        """
        print(f"Spin-stabilizing: h = {h}, Norm of angular momentum h_norm = {h_norm}")
        print("torque command: ", crossproduct(u) @ magnetic_field)
        angle_sun_h = np.arccos(np.clip(np.dot(sun_vector, h / h_norm), -1.0, 1.0))
        print("Angle between sun vector and angular momentum direction (degrees):", np.degrees(angle_sun_h))
        print("Angle between sun vector and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(sun_vector, I_min_direction))))
        print("Angle between angular momentum and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(h/h_norm, I_min_direction))))
        """
        return u.reshape(3, 1)

class BcrossController():
    def __init__(
        self,
        G_mtb: np.ndarray,
        b_dot_gain: float,
        Magnetorquers: list,
        inertia_tensor: np.ndarray,
        ref_angular_velocity: float,
    ) -> None:
        
        self.k = np.array(b_dot_gain)
        max_moms = np.zeros(len(Magnetorquers))
        for i, mtb in enumerate(Magnetorquers):
            max_moms[i] = mtb.max_dipole_moment
        self.ubm = np.min(np.abs(G_mtb).T @ max_moms)
        self.lbm = -self.ubm
        self.J = inertia_tensor 
        
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction
                
        self.ref_angular_velocity = self.I_min_direction * np.deg2rad(ref_angular_velocity)
 
    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        B-cross law: https://arc.aiaa.org/doi/epdf/10.2514/1.53074
        """
        refomega_norm = np.linalg.norm(self.ref_angular_velocity)
        if np.linalg.norm(angular_velocity - self.ref_angular_velocity) / refomega_norm >= np.deg2rad(5):
            magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
            unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)
            m_cmd = -self.k * np.cross(unit_magnetic_field, angular_velocity - self.ref_angular_velocity).reshape(3,1)
        else:
            m_cmd = np.zeros((3,1))
        return np.clip(a=m_cmd, a_min=self.lbm, a_max=self.ubm)
