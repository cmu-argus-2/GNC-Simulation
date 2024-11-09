
import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *
from FSW.controllers.ControllerAlgorithm import ControllerAlgorithm  

class BaselineNadirPointingController(ControllerAlgorithm):
    def __init__(
        self,
        Magnetorquers: list,
        ReactionWheels: list,
        params: dict,        
    ):
        super().__init__(Magnetorquers, ReactionWheels, params)  # Call the parent class's __init__ method

        self.kdetumb = np.array(params["bcrossgain"])
        self.k_mtb_att = np.array(params["mtb_att_feedback_gains"])
        
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction 
        # self.I_min_direction = self.G_rw_b.flatten() 
        self.target_angular_velocity = np.array(params["tgt_ss_ang_vel"])
        self.ref_angular_velocity = self.I_min_direction * np.deg2rad(self.target_angular_velocity)
        self.h_tgt = self.J @ self.ref_angular_velocity
        self.h_tgt_norm = np.linalg.norm(self.h_tgt)

        self.nadir_cam_dir = np.array(params["nadir_cam_dir"])
        self.target_rw_ang_vel = np.array(params["nom_rw_ang_vel"])
        self.k_mtb_rw = np.array(params["rw_vel_gain"])
        self.k_rw_att = np.array(params["rw_att_feedback_gains"])

    # TODO: maybe move this to a utility file
    def remove_projection_from_vector(self, 
                                      vector: np.ndarray, 
                                      projection: np.ndarray) -> np.ndarray:
        projection = projection / np.linalg.norm(projection)
        return vector - np.dot(vector, projection) * projection

    def get_dipole_moment_and_rw_torque_command(
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

        rw_ang_vel = est_ctrl_states[Idx["X"]["RW_SPEED"]]

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
                          (np.arccos(np.clip(np.dot(self.I_min_direction, h / h_norm), -1.0, 1.0)) <= np.deg2rad(15))
        # if angular velocity low enough, then its in nadir pointing and dont need to spin stabilize
        spin_stabilized = spin_stabilized or (np.linalg.norm(angular_velocity) <= np.deg2rad(1))
        # orbit_pointing = (np.linalg.norm(orbit_vector-(h/h_norm))<= np.deg2rad(10))
        orbit_pointing = (np.linalg.norm(orbit_vector-self.G_rw_b.flatten())<= np.deg2rad(10))
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        
        unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)

        if not spin_stabilized:
            u_mtb = self.kdetumb * np.cross(unit_magnetic_field, self.ref_angular_velocity - angular_velocity).reshape(3,1)
        elif not orbit_pointing: # pointing rw to orbit normal
            Bhat = crossproduct(magnetic_field)
            Bhat_pseudo_inv = Bhat.T / (magnetic_field_norm ** 2)

            err_vec = np.hstack((-self.G_rw_b.flatten()-orbit_vector, self.ref_angular_velocity - angular_velocity))
            u_mtb = Bhat_pseudo_inv @ self.k_mtb_att @ err_vec
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
            err_vec = np.hstack((-self.G_rw_b.flatten()-orbit_vector, angular_velocity))
            u_mtb = Bhat_pseudo_inv @ self.k_mtb_att @ err_vec
            u_mtb = np.clip(a=u_mtb, a_min=self.lbmtb, a_max=self.ubmtb)
            # 2. rw spin stabilization
            err_rw = self.target_rw_ang_vel - rw_ang_vel
            t_mtb_rw = Bhat_pseudo_inv @ self.G_rw_b * self.k_mtb_rw * err_rw
            ff_rw_tq = np.clip(a=u_mtb + t_mtb_rw.flatten(), a_min=self.lbmtb, a_max=self.ubmtb) - u_mtb
            u_mtb += ff_rw_tq
            # feedforward magnetorquer feedforward toset rw ang velocity
            ff_rw_tq = np.linalg.norm(Bhat @ t_mtb_rw)
            # ff_rw_tq = t_mtb_rw - (np.dot(t_mtb_rw, magnetic_field) / magnetic_field_norm**2) * magnetic_field
            # 3. rw attitude control
            tgt_rw_dir = self.remove_projection_from_vector(nadir_vector, self.G_rw_b.flatten())
            cur_rw_dir = self.remove_projection_from_vector(self.nadir_cam_dir, self.G_rw_b.flatten())
            tgt_rw_dir =  tgt_rw_dir / np.linalg.norm(tgt_rw_dir)
            cur_rw_dir =  cur_rw_dir / np.linalg.norm(cur_rw_dir)
            angle_norm = np.abs(np.arccos(np.clip(np.dot(tgt_rw_dir, cur_rw_dir), -1.0, 1.0)))
            angle_sign = np.sign(np.dot(np.cross(self.G_rw_b.flatten(),cur_rw_dir), tgt_rw_dir))
            angle_nadir = angle_sign * angle_norm
            err_att = np.hstack((angle_nadir, -self.G_rw_b.flatten() @ angular_velocity))
            u_rw = self.k_rw_att @ err_att - ff_rw_tq
       
        u_mtb = np.clip(a=u_mtb, a_min=self.lbmtb, a_max=self.ubmtb)
        u_rw = np.clip(a=u_rw, a_min=self.lbrw, a_max=self.ubrw)
        # clip according to rw angular velocity
        
        """
        print(f"Spin-stabilizing: h = {h}, Norm of angular momentum h_norm = {h_norm}")
        print("torque command: ", crossproduct(u) @ magnetic_field)
        angle_sun_h = np.arccos(np.clip(np.dot(sun_vector, h / h_norm), -1.0, 1.0))
        print("Angle between sun vector and angular momentum direction (degrees):", np.degrees(angle_sun_h))
        print("Angle between sun vector and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(sun_vector, I_min_direction))))
        print("Angle between angular momentum and I_min_direction (degrees):", np.degrees(np.arccos(np.dot(h/h_norm, I_min_direction))))
        """
        return u_mtb.reshape(3, 1), u_rw