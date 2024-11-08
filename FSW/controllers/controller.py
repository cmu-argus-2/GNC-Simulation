import configparser
import numpy as np
from world.math.quaternions import *
from FSW.controllers.ControllerAlgorithm import *
from FSW.controllers.BcrossController import BcrossController
from FSW.controllers.LyapBasedSunPointingController import LyapBasedSunPointingController
from FSW.controllers.BaselineSunPointingController import BaselineSunPointingController
from FSW.controllers.BaselineNadirPointingController import BaselineNadirPointingController

class Controller:
    def __init__(self, config, Magnetorquers, ReactionWheels, Idx) -> None:
        self.config = config
        self.pointing_mode    = config["pointing_mode"]
        self.pointing_target  = config["pointing_target"]
        self.controller_algo  = config["algorithm"]
        self.tgt_ang_vel      = np.array(config["tgt_ss_ang_vel"], dtype=np.float64)
        self.est_world_states = None
        self.mtb              = Magnetorquers
        self.G_mtb_b = np.array(config["mtb_orientation"]).reshape(config["N_mtb"], 3)
        self.G_rw_b  = np.array(config["rw_orientation"]).reshape(config["N_rw"], 3)
        self.inertia = np.array(config["inertia"]).reshape(3,3)
    
        self.controller_algorithm = None
        if self.controller_algo == "Bcross":
            self.controller_algorithm = BcrossController(
                                                    Magnetorquers,
                                                    ReactionWheels,
                                                    config)
        elif self.controller_algo == "Lyapunov":
            self.controller_algorithm = LyapBasedSunPointingController(
                                                    Magnetorquers,
                                                    ReactionWheels,
                                                    config)
        elif self.controller_algo == "BaseSP":
            self.controller_algorithm = BaselineSunPointingController(
                                                    Magnetorquers,
                                                    ReactionWheels,
                                                    config)
        elif self.controller_algo == "BaseNP":
            self.controller_algorithm = BaselineNadirPointingController(
                                                    Magnetorquers,
                                                    ReactionWheels,
                                                    config)
        else:
            raise ValueError(f"Unrecognized controller algorithm: {self.controller_algo}")
  
        self.allocation_mat = np.zeros((Idx["NU"],3))
       
    def _load_gains(self):
        gains = {}
        if 'Gains' in self.config:
            for key in self.config['Gains']:
                gains[key] = float(self.config['Gains'][key])
        return gains
    
    
    def allocate_torque(self, state, mtb_torque_cmd, rw_torque_cmd, Idx):
        # Placeholder for actuator management
        # if torque = B @ actuator_cmd
        # actuator_cmd = Binv @ torque_cmd
        
     
        self.allocation_mat = np.zeros((Idx["NU"],3))
        self.allocation_mat[Idx["U"]["MTB_TORQUE"],:] = np.linalg.pinv(self.G_mtb_b.T)
                
        # np.linalg.pinv(B_mat)
        actuator_cmd = np.zeros((Idx["NU"],1))
        actuator_cmd[Idx["U"]["MTB_TORQUE"]] = np.linalg.pinv(self.G_mtb_b.T) @ mtb_torque_cmd
        if rw_torque_cmd:
            actuator_cmd[Idx["U"]["RW_TORQUE"]] = rw_torque_cmd # only one, bypass allocation matrix
        # np.linalg.pinv(self.G_rw_b.T) @ rw_torque_cmd
        # Normalize columns of the allocation matrix
        # col_norms = np.linalg.norm(allocation_mat, axis=0)
        # allocation_mat = allocation_mat / col_norms
        
        return actuator_cmd
    
    def run(self, date, est_world_states, Idx):
        
        self.est_world_states = est_world_states
       
        # feedforward and feedback controller
        mtb_torque_cmd, rw_torque_cmd = self.controller_algorithm.get_dipole_moment_and_rw_torque_command(est_world_states, Idx)
        
        # actuator management function
        actuator_cmd = self.allocate_torque(self.est_world_states, mtb_torque_cmd, rw_torque_cmd, Idx)

        # convert dipole moment to voltage command     
        volt_cmd = np.zeros((Idx["NU"],))
        for i in range(len(self.mtb)):
            volt_cmd[Idx["U"]["MTB_TORQUE"]][i] = self.mtb[i].convert_dipole_moment_to_voltage(actuator_cmd[Idx["U"]["MTB_TORQUE"]][i])
        # [To determine:] RW interface. Baseline torque/current control
        volt_cmd[Idx["U"]["RW_TORQUE"]] = actuator_cmd[Idx["U"]["RW_TORQUE"]]

        return volt_cmd.flatten()

        