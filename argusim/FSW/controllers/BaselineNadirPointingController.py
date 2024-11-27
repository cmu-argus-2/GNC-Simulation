
import numpy as np
from argusim.actuators import Magnetorquer
from argusim.world.math.quaternions import *
from argusim.FSW.controllers.ControllerAlgorithm import ControllerAlgorithm  

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

        # target rotation matrix
        # Get the quaternion in the body frame that points G_rw to the orbit_vector
        q_tgt = quat_from_two_vectors(orbit_vector, nadir_vector)

        # Rotate nadir_cam_dir using q_rw_to_orbit
        q_ref = quat_from_two_vectors(self.G_rw_b.flatten(), self.nadir_cam_dir)

        # Combine the two quaternions to get the final target quaternion
        q_target = hamiltonproduct(q_tgt, quatconj(q_ref))
        axis, angle = quat_to_axis_angle(q_target)
        err_att = 0*axis * angle
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        
        # target quaternion from orbit_vector and nadir_vector
        # nadir_cam_dir points t onadir_vector in target orientation

        """naive pd control
        Bhat = crossproduct(magnetic_field)
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        Bhat_pseudo_inv = Bhat.T / (magnetic_field_norm ** 2)
        Bmat = np.zeros((3,4))
        Bmat[:3,:3] = Bhat.T 
        Bmat[:3,-1] = self.G_rw_b.flatten()

        alloc_mat = np.zeros((4,3))
        if np.abs(np.dot(magnetic_field, self.G_rw_b.flatten())) / magnetic_field_norm < 1e-6: 
            # [TODO]: fix this. No control over rw speed and another axis
            alloc_mat[:,:3] = np.linalg.pinv(Bmat[:3,:])
        else:
            alloc_mat = np.linalg.pinv(Bmat)
        # attitude control torque
        u = alloc_mat @ self.k_mtb_att @ np.hstack((err_att, -angular_velocity))

        # rw control torque
        # u[:3] -= Bhat_pseudo_inv @ self.G_rw_b.flatten() *  self.k_mtb_rw * (self.target_rw_ang_vel - rw_ang_vel)
        u_mtb = self.clip_total_dipole_moment(u[:3])
        u_rw = self.clip_rw_torque(u[3], rw_ang_vel)
        """
        """
        # clipping the torques
        uatt[:3] = self.clip_total_dipole_moment(uatt[:3])
        uatt[3] = self.clip_rw_torque(uatt[3], rw_ang_vel)

        u_wrw[:3] = self.clip_total_dipole_moment(u_wrw[:3] + uatt[:3]) - uatt[:3]
        u_wrw[3] = self.clip_rw_torque(u_wrw[3] + uatt[3], rw_ang_vel) - uatt[3]

        max_rw_torque = min(np.linalg.norm(Bmat[3, :3] @ u_wrw[:3]), np.linalg.norm(Bmat[3, -1] * u_wrw[3]))
        u_wrw = alloc_mat[:,-1] * max_rw_torque * np.sign(self.target_rw_ang_vel - rw_ang_vel)
        
        u_mtb = uatt[:3] + u_wrw[:3]
        u_rw  = uatt[3] + u_wrw[3]
        """

        """"""
        u_mtb = np.zeros(3)
        u_rw  = np.zeros(1)
        # angular momentum in a cone around the direction to be pointed
        spin_stabilized = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))

        #spin_stabilized = (h_norm <= (1.0 + np.deg2rad(15)) * self.h_tgt_norm) and \
        #                  (np.arccos(np.clip(np.dot(self.I_min_direction, h / h_norm), -1.0, 1.0)) <= np.deg2rad(15))
        # if angular velocity low enough, then its in nadir pointing and dont need to spin stabilize
        #spin_stabilized = spin_stabilized or (np.linalg.norm(angular_velocity) <= np.deg2rad(1))
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
            # err_vec = np.hstack((err_att, -angular_velocity))
            
            u_mtb = Bhat_pseudo_inv @ self.k_mtb_att @ err_vec
        else:
            # rw att control

            # magnetorquer desaturation
            """

            # nadir pointing controller
            Bhat = crossproduct(magnetic_field)
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
            """

        u_mtb = self.clip_total_dipole_moment(u_mtb)
        u_rw = self.clip_rw_torque(u_rw, rw_ang_vel)
        # clip according to rw angular velocity
        
        return u_mtb.reshape(3, 1), u_rw