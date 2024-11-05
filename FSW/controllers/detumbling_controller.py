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
        est_ctrl_states: np.ndarray,
        Idx: dict,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        Re2b = quatrotation(est_ctrl_states[Idx["X"]["QUAT"]]).T
        magnetic_field   = Re2b @ est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        sun_vector       = Re2b @ est_ctrl_states[Idx["X"]["SUN_POS"]]
        sun_vector       = sun_vector / np.linalg.norm(sun_vector)

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
        est_ctrl_states: np.ndarray,
        Idx: dict,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        Re2b = quatrotation(est_ctrl_states[Idx["X"]["QUAT"]]).T
        magnetic_field   = Re2b @ est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        sun_vector       = Re2b @ est_ctrl_states[Idx["X"]["SUN_POS"]]
        sun_vector       = sun_vector / np.linalg.norm(sun_vector)

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

class BaselineNadirPointingController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        G_mtb: np.ndarray,
        b_cross_gain: float,
        mtb_pd_gain: np.ndarray,
        rw_pd_gain: np.ndarray,
        rw_vel_gain: float,
        target_angular_velocity: np.ndarray,
        target_rw_ang_vel: np.ndarray,
        nadir_cam_dir: np.ndarray,
        Magnetorquers: list,
        ReactionWheels: list,
    ) -> None:
        # [TODO] max moment should be defined per axis based on all magnetorquers, not just first
        max_moms = np.zeros(len(Magnetorquers))
        for i, mtb in enumerate(Magnetorquers):
            max_moms[i] = mtb.max_dipole_moment
        
        self.ubmtb = np.min(np.abs(G_mtb).T @ max_moms)
        self.lbmtb = -self.ubmtb

        self.ubrw = np.array([rw.max_torque for rw in ReactionWheels])
        self.lbrw = -self.ubrw
        self.G_rw_b  = np.array([rw.G_rw_b for rw in ReactionWheels]).reshape(3, -1)

        self.kdetumb = np.array(b_cross_gain)
        self.k_mtb_att = np.array(mtb_pd_gain)
        
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

        self.nadir_cam_dir = nadir_cam_dir
        self.target_rw_ang_vel = target_rw_ang_vel
        self.k_mtb_rw = rw_vel_gain
        self.k_rw_att = np.array(rw_pd_gain)

    def remove_projection_from_vector(vector, projection):
        projection = projection / np.linalg.norm(projection)
        return vector - np.dot(vector, projection) * projection

    def get_dipole_moment_command_and_rw_torque(
        self,
        est_ctrl_states: np.ndarray,
        Idx: dict,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """

        """ 
        q_ref = ref_ctrl_states[:4]
        q_est = est_ctrl_states[Idx["X"]["QUAT"]]
        q_err = hamiltonproduct(q_ref, q_inv(q_est))
        err_vec = np.hstack((q_err[1:], ref_ctrl_states[4:7] - est_ctrl_states[Idx["X"]["ANG_VEL"]]))
        
        torque = ref_torque + self.feedback_gains @ err_vec
        return torque
        """ 
        Re2b = quatrotation(est_ctrl_states[Idx["X"]["QUAT"]]).T
        magnetic_field   = Re2b @ est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        sun_vector       = Re2b @ est_ctrl_states[Idx["X"]["SUN_POS"]]
        sun_vector       = sun_vector / np.linalg.norm(sun_vector)

        rw_ang_vel = est_ctrl_states[Idx["X"]["ANG_VEL"]]

        eci_pos = est_ctrl_states[Idx["X"]["ECI_POS"]]
        eci_vel = est_ctrl_states[Idx["X"]["ECI_VEL"]]
        orbit_vector = Re2b @ np.cross(eci_pos, eci_vel)
        orbit_vector = orbit_vector / np.linalg.norm(orbit_vector)
        if np.dot(orbit_vector,sun_vector) < 0:
            orbit_vector = -orbit_vector
        nadir_vector = Re2b @ (-eci_pos)
        nadir_vector = nadir_vector / np.linalg.norm(nadir_vector)

        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        
        # target quaternion from orbit_vector and nadir_vector
        # nadir_cam_dir points t onadir_vector in target orientatio

        u_mtb = np.zeros(3)
        u_rw  = np.zeros(1)
         
        # angular momentum in a cone around the direction to be pointed
        spin_stabilized = (h_norm <= (1.0 + np.deg2rad(15)) * self.h_tgt_norm) and \
                          (np.arccos(np.clip(np.dot(self.G_rw_b.flatten(), h / h_norm), -1.0, 1.0)) <= np.deg2rad(15))
   
        orbit_pointing = (np.linalg.norm(orbit_vector-(h/h_norm))<= np.deg2rad(10))
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        
        unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)

        if not spin_stabilized:
            u_mtb = self.kdetumb * np.cross(unit_magnetic_field, self.ref_angular_velocity - angular_velocity).reshape(3,1)
        elif not orbit_pointing: # pointing rw to orbit normal
            Bhat = crossproduct(magnetic_field)
            Bhat_pseudo_inv = Bhat.T / (magnetic_field_norm ** 2)

            err_vec = np.hstack((-self.G_rw_b-orbit_vector, self.ref_angular_velocity - angular_velocity))
            u_mtb = Bhat_pseudo_inv @ self.k @ err_vec
        else:
            # nadir pointing controller
            Bhat = crossproduct(magnetic_field)
            """
            U, S, V = np.linalg.svd(Bhat)
            idx = np.where(S > 1e-6 * np.max(S))[0]
            S_inv = np.diag([1/s if s > 1e-10 else 0 for s in S])
            Bhat_pseudo_inv = V.T @ S_inv @ U.T"""
            Bhat_pseudo_inv = Bhat.T / (magnetic_field_norm ** 2)
            # 1. mtb nutation control
            err_vec = np.hstack((-self.G_rw_b-orbit_vector, angular_velocity))
            u_mtb = Bhat_pseudo_inv @ self.k_mtb_att @ err_vec
            u_mtb = np.clip(a=u_mtb, a_min=self.lbm, a_max=self.ubm)
            # 2. rw spin stabilization
            err_rw = self.target_rw_ang_vel - rw_ang_vel
            t_mtb_rw = Bhat_pseudo_inv @ self.G_rw_b @ self.k_mtb_rw @ err_rw
            ff_rw_tq = np.clip(a=u_mtb + t_mtb_rw, a_min=self.lbm, a_max=self.ubm) - u_mtb
            u_mtb += ff_rw_tq
            # feedforward magnetorquer feedforward toset rw ang velocity
            ff_rw_tq = np.linalg.norm(Bhat @ t_mtb_rw)
            # ff_rw_tq = t_mtb_rw - (np.dot(t_mtb_rw, magnetic_field) / magnetic_field_norm**2) * magnetic_field
            # 3. rw attitude control
            tgt_rw_dir = self.remove_projection_from_vector(nadir_vector, self.G_rw_b)
            cur_rw_dir = self.remove_projection_from_vector(self.nadir_cam_dir, self.G_rw_b)
            angle_norm = np.abs(np.arccos(np.clip(np.dot(tgt_rw_dir, cur_rw_dir), -1.0, 1.0)))
            angle_sign = np.sign(np.dot(np.cross(self.G_rw_b,cur_rw_dir), tgt_rw_dir))
            angle_nadir = angle_sign * angle_norm
            err_att = np.hstack((angle_nadir, -angular_velocity))
            u_rw = self.k_rw_att @ err_att - ff_rw_tq
       
        u_mtb = np.clip(a=u_mtb, a_min=self.lbmtb, a_max=self.ubmtb)
        u_rw = np.clip(a=u_rw, a_min=self.lbrw, a_max=self.ubrw)
        """
        print(f"Spin-stabilizing: h = {h}, Norm of angular momentum h_norm = {h_norm}")
        print("torque command: ", crossproduct(u) @ magnetic_field)
        angle_sun_h = np.arccos(np.clip(np.dot(sun_vector, h / h_norm), -1.0, 1.0))
        print("Angle between sun vector and angular momentum direction (degrees):", np.degrees(angle_sun_h))
        print("Angle between sun vector and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(sun_vector, I_min_direction))))
        print("Angle between angular momentum and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(h/h_norm, I_min_direction))))
        """
        return u_mtb.reshape(3, 1), u_rw

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
        est_ctrl_states: np.ndarray,
        Idx: dict,
    ) -> np.ndarray:
        """
        B-cross law: https://arc.aiaa.org/doi/epdf/10.2514/1.53074
        """
        Re2b = quatrotation(est_ctrl_states[Idx["X"]["QUAT"]]).T
        magnetic_field = Re2b @ est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        refomega_norm = np.linalg.norm(self.ref_angular_velocity)
        if np.linalg.norm(angular_velocity - self.ref_angular_velocity) / refomega_norm >= np.deg2rad(5):
            magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
            unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)
            m_cmd = -self.k * np.cross(unit_magnetic_field, angular_velocity - self.ref_angular_velocity).reshape(3,1)
        else:
            m_cmd = np.zeros((3,1))
        return np.clip(a=m_cmd, a_min=self.lbm, a_max=self.ubm)
