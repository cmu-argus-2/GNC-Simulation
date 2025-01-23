
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
        self.target_angular_velocity = np.deg2rad(np.array(params["tgt_ss_ang_vel"]))
        self.ref_angular_velocity = self.I_min_direction * self.target_angular_velocity
        self.h_tgt = self.J @ self.ref_angular_velocity
        self.h_tgt_norm = np.linalg.norm(self.h_tgt)

        self.nadir_cam_dir = np.array(params["nadir_cam_dir"])
        self.target_rw_ang_vel = np.deg2rad(np.array(params["nom_rw_ang_vel"]))
        self.rw_J =np.array(params["reaction_wheels"]["I_rw"])
        self.rw_h_tgt = self.rw_J * self.target_rw_ang_vel
        self.k_mtb_rw = np.array(params["rw_vel_gain"])
        self.k_rw_att = np.array(params["rw_att_feedback_gains"])
        self.orb_pointed = False

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
        att_quat         = est_ctrl_states[Idx["X"]["QUAT"]]
        magnetic_field   = est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        sun_vector       = est_ctrl_states[Idx["X"]["SUN_POS"]]
        zenith_vector    = est_ctrl_states[Idx["X"]["ECI_POS"]]
        nadir_vector     = -zenith_vector
        cross_vector     = est_ctrl_states[Idx["X"]["ECI_VEL"]]
        rw_ang_vel       = est_ctrl_states[Idx["X"]["RW_SPEED"]]
        orbit_vector     = np.cross(zenith_vector, cross_vector)
        orbit_vector     = orbit_vector / np.linalg.norm(orbit_vector)
        if np.dot(orbit_vector,sun_vector) < 0:
            orbit_vector = -orbit_vector

        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        u_mtb = np.zeros(3)
        u_rw = 0.0
        angle_ss = (np.arccos(np.dot(h/h_norm, self.I_min_direction)) <= np.deg2rad(15))
        norm_ss  = (h_norm <= self.h_tgt_norm  * (1.0+np.deg2rad(15)))
        spin_stabilized     = angle_ss and norm_ss
        # spin_stabilized     = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))
        fine_orb_pointing   = (np.linalg.norm(orbit_vector-(h/self.h_tgt_norm))<= np.deg2rad(10))
        coarse_orb_pointing = (np.linalg.norm(orbit_vector-(h/self.h_tgt_norm))<= np.deg2rad(10))
        # fine_orb_pointing2  = (np.linalg.norm(orbit_vector-(self.G_rw_b.flatten()))<= np.deg2rad(30))
        if not coarse_orb_pointing or not spin_stabilized:  
            self.nadir_pointed = False
            orb_pointing = fine_orb_pointing
        if self.orb_pointed:
            orb_pointing = coarse_orb_pointing
        else:
            orb_pointing = fine_orb_pointing

        if not spin_stabilized and not self.orb_pointed:
            # u = crossproduct(magnetic_field) @ (self.I_min_direction - (h/self.h_tgt_norm))
            u_mtb = crossproduct(magnetic_field) @ (self.I_min_direction*self.h_tgt_norm - h) * self.kdetumb

            if np.linalg.norm(u_mtb) == 0:
                u_mtb = np.zeros(3)
            else:
                u_mtb = self.ubmtb * np.tanh(u_mtb) 
        
        elif not orb_pointing and not self.orb_pointed:
            u_mtb = crossproduct(magnetic_field) @ (orbit_vector - (h/self.h_tgt_norm)) # (h/self.h_tgt_norm))
            if np.linalg.norm(u_mtb) == 0:
                u_mtb = np.zeros(3)
            else:
                u_mtb = self.ubmtb * u_mtb/np.linalg.norm(u_mtb) 
                # element-wise division
                # u = self.ubmtb * np.sign(u) 
        else:
            self.orb_pointed = True
            """
            q_tgt = quat_from_two_vectors(orbit_vector, nadir_vector)

            # Rotate nadir_cam_dir using q_rw_to_orbit
            q_ref = quat_from_two_vectors(self.G_rw_b.flatten(), self.nadir_cam_dir)

            # Combine the two quaternions to get the final target quaternion
            q_target = hamiltonproduct(q_tgt, quatconj(q_ref))

            axis, angle = quat_to_axis_angle(q_target)
            """
            Rtgt = np.vstack((orbit_vector,nadir_vector, np.cross(orbit_vector, nadir_vector))).T
            R_ref = np.vstack((self.G_rw_b.flatten(), self.nadir_cam_dir, np.cross(self.G_rw_b.flatten(), self.nadir_cam_dir))).T
            R = Rtgt @ np.linalg.pinv(R_ref)
            q =  rotmat2quat(R)
            axis, angle = quat_to_axis_angle(q)
            
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < -np.pi:
                angle += 2 * np.pi
            err_att = axis * angle
            # PD controller
            Bhat = crossproduct(magnetic_field)
            magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
            magfield_hat = magnetic_field / magnetic_field_norm
            Bhat_pseudo_inv = Bhat / (magnetic_field_norm ** 2)
            uatt = self.k_mtb_att @ np.hstack((-self.J @ err_att, - self.J @ angular_velocity))
            u_wrw = self.k_mtb_rw * self.rw_J * (self.target_rw_ang_vel - rw_ang_vel)
            cross_prod = np.linalg.norm(np.cross(magfield_hat, self.G_rw_b.flatten()))
            u_mtb = Bhat_pseudo_inv @ np.hstack((uatt[:2],-u_wrw))
            u_rw  = uatt[2] # + cross_prod * u_wrw
            """
            Bmat = np.zeros((4,4))
            Bmat[:3,:3] = Bhat_pseudo_inv
            Bmat[:3,-1] = self.G_rw_b.flatten()
            
            Bmat[-1,:3] = Bmat[:3,:] @ np.hstack((-self.G_rw_b.flatten(),cross_prod))

            alloc_mat = np.zeros((4,3))
            if np.abs(np.dot(magnetic_field, self.G_rw_b.flatten())) / magnetic_field_norm < 1e-6: 
                # [TODO]: fix this. No control over rw speed and another axis
                alloc_mat[:,:3] = np.linalg.pinv(Bmat[:3,:])
            else:
                alloc_mat = np.linalg.pinv(Bmat)
            # attitude control torque
            uatt = alloc_mat[:,:3] @ self.k_mtb_att @ np.hstack((err_att, -angular_velocity))

            # clipping the torques
            uatt[:3] = self.clip_total_dipole_moment(uatt[:3])
            uatt[3] = self.clip_rw_torque(uatt[3], rw_ang_vel)
            rwlloc = Bmat @ np.hstack((-self.G_rw_b.flatten(),1))
            rwlloc = rwlloc / (np.linalg.norm(rwlloc) ** 2)
            u_wrw = alloc_mat[:,3] * self.k_mtb_rw * self.rw_J * (self.target_rw_ang_vel - rw_ang_vel)

            u_mtb = u_wrw[:3]  + uatt[:3]
            u_rw  = u_wrw[3]   + uatt[3]
            """


        u_mtb = self.clip_total_dipole_moment(u_mtb)
        u_rw = self.clip_rw_torque(u_rw, rw_ang_vel)
        # clip according to rw angular velocity
        
        return u_mtb.reshape(3, 1), u_rw