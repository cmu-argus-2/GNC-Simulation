
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
            # Re2b = self.attitude_ekf.get_ECI_R_b().as_matrix()
            vtgt1 = orbit_vector / np.linalg.norm(orbit_vector)
            vtgt2 = np.cross(orbit_vector, nadir_vector) / np.linalg.norm(np.cross(orbit_vector, nadir_vector))
            vtgt3 = np.cross(vtgt1, vtgt2) / np.linalg.norm(np.cross(vtgt1, vtgt2))
            Rtgt = np.vstack((vtgt1,vtgt2, vtgt3)).T
            vref1 = self.G_rw_b.flatten() / np.linalg.norm(self.G_rw_b.flatten())
            vref2 = np.cross(self.G_rw_b.flatten(), self.nadir_cam_dir) / np.linalg.norm(np.cross(self.G_rw_b.flatten(), self.nadir_cam_dir))
            vref3 = np.cross(vref1, vref2) / np.linalg.norm(np.cross(vref1, vref2))
            R_ref = np.vstack((vref1,vref2, vref3)).T
            
            R = Rtgt @ R_ref.T
            q =  rotmat2quat(R)
            axis, angle = quat_to_axis_angle(q)
            err_att = axis * angle
            for i in range(3):
                if err_att[i] > np.pi:
                    err_att[i] -= 2 * np.pi
                elif err_att[i] < -np.pi:
                    err_att[i] += 2 * np.pi
            # PD controller
            Bhat = crossproduct(magnetic_field)
            magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
            magfield_hat = magnetic_field / magnetic_field_norm
            Bhat_pseudo_inv = Bhat / (magnetic_field_norm ** 2)
            ref_ang_vel = orbit_vector*2*np.pi/86400.0
            if np.abs(rw_ang_vel) > 10*2*np.pi:
                Kd = self.rw_J *rw_ang_vel / np.sqrt(self.J[0,0]*self.J[1,1])
            else:
                Kd = 0.0
            # if angle grows too large, torque closer to orbit normal
            Kdrw  = 0.5 * self.J[2,2]
            Katt = np.zeros((3,6))
            Katt[2,2] = (Kdrw ** 2) / (2.0 * self.J[2,2])
            Katt[0,3] = Kd
            Katt[1,4] = Kd
            Katt[2,5] = Kdrw
            # repoint if strayed too far
            if np.linalg.norm(err_att[:2]) > np.deg2rad(10):
                Kp = Kd ** 2 / (2 * self.J[0,0])
                Katt[0,0] = Kp
                Katt[1,1] = Kp
            
            uatt = Katt @ np.hstack((err_att,ref_ang_vel - angular_velocity))
            if np.linalg.norm(self.target_rw_ang_vel - rw_ang_vel) >= 3*2*np.pi:
                u_wrw = self.k_mtb_rw * self.rw_J * (self.target_rw_ang_vel - rw_ang_vel)
            else:
                u_wrw = 0.0
            
            cross_prod = np.linalg.norm(np.cross(magfield_hat, self.G_rw_b.flatten()))
            
            # if np.linalg.norm(self.target_rw_ang_vel - rw_ang_vel) <= 3*2*np.pi:
            u_mtb = Bhat_pseudo_inv @ np.hstack((uatt[:2],u_wrw))
            u_rw  = -uatt[2] + cross_prod * u_wrw
            

        u_mtb = self.clip_total_dipole_moment(u_mtb)
        u_rw = self.clip_rw_torque(u_rw, rw_ang_vel)
        # clip according to rw angular velocity
        
        return u_mtb.reshape(3, 1), u_rw