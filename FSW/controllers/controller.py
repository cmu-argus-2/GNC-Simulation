import configparser
import numpy as np
from world.math.quaternions import *
from FSW.controllers.detumbling_controller import *

class Controller:
    def __init__(self, config, Idx) -> None:
        self.config = config
        self.pointing_mode    = config["mission"]["pointing_mode"]
        self.controller_algo  = config["controller"]["algorithm"]
        self.feedback_gains   = np.array(config["controller"]["state_feedback_gains"], dtype=np.float64)
        self.est_world_states = None
        self.Bcrossctr  = BcrossController(config["controller"]["bcrossgain"])
        self.G_mtb_b = np.array(config["satellite"]["mtb_orientation"])
        self.G_rw_b  = np.array(config["satellite"]["rw_orientation"])
        self.allocation_mat = np.zeros((Idx["NU"],3))
       
    def _load_gains(self):
        gains = {}
        if 'Gains' in self.config:
            for key in self.config['Gains']:
                gains[key] = float(self.config['Gains'][key])
        return gains
    
    # define feedforward torque and state profile
    def define_att_profile(self, date, pointing_mode, est_world_states):
        ref_ctrl_states = np.zeros((7,))
        if pointing_mode == "Sun":
            # Define reference torque as zero
            ref_torque = np.zeros((3,))
            
            # Define reference control states with constant estimated sun direction
            # quaternion defining sun pointing direction
            # +z pointing to self.state[13:16]
            # +x cross product of +z and [0,0,1]
            # +y completes the right handed coordinate system
            z_body = -est_world_states[13:16] / np.linalg.norm(est_world_states[13:16])
            x_body = np.cross(z_body, np.array([0,0,1]))
            y_body = np.cross(z_body, x_body)
            Rot_mat = np.vstack((x_body, y_body, z_body)).T
            q_ref = rotmat2quat(Rot_mat)
            
            ref_ctrl_states[:4] = q_ref
            # reference angular velocity (approx zero)
            omega_ref  = np.zeros((3,))
            
            ref_ctrl_states[4:7] = omega_ref
        
        elif pointing_mode == "Nadir":
            att_states = est_world_states[6:13]
            # Default values if not sun pointing
            ref_ctrl_states = np.zeros((19,))
            ref_torque = np.zeros((3,))
        elif  pointing_mode == "detumble":
            # Default values if not sun pointing
            ref_ctrl_states = np.zeros((19,))
            ref_torque = np.zeros((3,))
        else:
            raise ValueError(f"Unrecognized pointing mode: {pointing_mode}")
            
        return ref_torque, ref_ctrl_states
    
    def get_torque(self, date, ref_torque, ref_ctrl_states, est_ctrl_states, Idx):
        # from the reference and estimated quaternions, get the error quaternion for control
        if self.controller_algo == "Bcross":
            Re2b = quatrotation(est_ctrl_states[Idx["X"]["QUAT"]]).T
            bfMAG_FIELD = Re2b @ est_ctrl_states[Idx["X"]["MAG_FIELD"]]
            return self.Bcrossctr.get_dipole_moment_command(bfMAG_FIELD, 
                                                            est_ctrl_states[Idx["X"]["ANG_VEL"]])
            
        q_ref = ref_ctrl_states[:4]
        q_est = est_ctrl_states[6:10]
        q_err = hamiltonproduct(q_ref, q_inv(q_est))
        err_vec = np.hstack((q_err[1:], ref_ctrl_states[4:7] - est_ctrl_states[10:13]))
        
        torque = ref_torque + self.feedback_gains @ err_vec
        return torque
    
    def allocate_torque(self, state, torque_cmd, Idx):
        # Placeholder for actuator management
        # if torque = B @ actuator_cmd
        # actuator_cmd = Binv @ torque_cmd
        
        B_mat = np.zeros((3, Idx["NU"]))
        B_mat[:,Idx["U"]["RW_TORQUE"]]  = self.G_rw_b.T
        mag_field = state[Idx["X"]["MAG_FIELD"]]
        Re2b = quatrotation(state[Idx["X"]["QUAT"]]).T
        bfMAG_FIELD = Re2b @ mag_field
        B_mat[:,Idx["U"]["MTB_TORQUE"]] = crossproduct(bfMAG_FIELD)  @ self.G_mtb_b
        
        self.allocation_mat = np.vstack((np.zeros((1, 3)), np.eye(3)))
        # np.linalg.pinv(B_mat)
        
        # Normalize columns of the allocation matrix
        # col_norms = np.linalg.norm(allocation_mat, axis=0)
        # allocation_mat = allocation_mat / col_norms
        
        actuator_cmd = self.allocation_mat @ torque_cmd
        return actuator_cmd
    
    def run(self, date, est_world_states, Idx):
        
        self.est_world_states = est_world_states
        # define slew profile
        ref_torque, ref_ctrl_states = self.define_att_profile(date, self.pointing_mode, est_world_states)
        # feedforward and feedback controller
        torque_cmd = self.get_torque(date, ref_torque, ref_ctrl_states, self.est_world_states, Idx)
        
        # actuator management function
        actuator_cmd = self.allocate_torque(self.est_world_states, torque_cmd, Idx)
                
        return actuator_cmd
        