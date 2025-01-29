import configparser
import numpy as np

from argusim.world.math.quaternions import *
from argusim.FSW.controllers.ControllerAlgorithm import *
from argusim.FSW.controllers.BcrossController import BcrossController
from argusim.FSW.controllers.LyapBasedSunPointingController import LyapBasedSunPointingController
from argusim.FSW.controllers.BaselineSunPointingController import BaselineSunPointingController
from argusim.FSW.controllers.BaselineNadirPointingController import BaselineNadirPointingController

class Controller:
    def __init__(self, 
                 config: dict, 
                 Magnetorquers: list, 
                 ReactionWheels: list, 
                 Idx: dict) -> None:
        self.config = config
        self.pointing_mode    = config["pointing_mode"]
        self.pointing_target  = config["pointing_target"]
        self.controller_algo  = config["algorithm"]
        self.tgt_ang_vel      = np.array(config["tgt_ss_ang_vel"], dtype=np.float64)
        self.est_world_states = None
        self.mtb              = Magnetorquers
        self.G_mtb_b = np.array(config["magnetorquers"]["mtb_orientation"]).reshape(config["magnetorquers"]["N_mtb"], 3)
        # normalize the rows
        self.G_mtb_b = self.G_mtb_b / np.linalg.norm(self.G_mtb_b, axis=1, keepdims=True)
        self.G_rw_b  = np.array(config["reaction_wheels"]["rw_orientation"]).reshape(config["reaction_wheels"]["N_rw"], 3)
        # control allocation matrix (only for magnetorquers)
        self.allocation_mat = np.zeros((Idx["NU"],3))
        self.allocation_mat[Idx["U"]["MTB_TORQUE"],:] = np.linalg.pinv(self.G_mtb_b.T)
        # no point allocating rw if its only one
        self.inertia = np.array(config["inertia"]["nominal_inertia"]).reshape(3,3)
    
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
        
        # debug flags
        # bypass controller and allocation matrix
        
        self.bypass_controller = config["debugFlags"]["bypass_controller"]
    
    def allocate_torque(self, 
                        mtb_torque_cmd: np.ndarray, 
                        rw_torque_cmd: np.ndarray, 
                        Idx: dict) -> np.ndarray:    
        """
        Allocates torque commands to the appropriate actuators.
        Args:
            mtb_torque_cmd (numpy.ndarray): Commanded torque for the magnetorquer bars (MTB).
            rw_torque_cmd (numpy.ndarray or None): Commanded torque for the reaction wheels (RW). If None, no torque is applied to the reaction wheels.
            Idx (dict): Dictionary containing indices for the actuators and their commands.
        Returns:
            numpy.ndarray: Actuator command array with allocated torques.
        """
        actuator_cmd = np.zeros((Idx["NU"],1))
        actuator_cmd[Idx["U"]["MTB_TORQUE"]] = \
                    self.allocation_mat[Idx["U"]["MTB_TORQUE"],:] @ mtb_torque_cmd
        if rw_torque_cmd:
            actuator_cmd[Idx["U"]["RW_TORQUE"]] = rw_torque_cmd # only one, bypass allocation matrix
        
        return actuator_cmd
    
    def run(self, est_world_states: np.ndarray, Idx: dict) -> np.ndarray:
        """
        Executes the control algorithm to generate actuator commands based on the estimated world states.
        Args:
            est_world_states (dict): A dictionary containing the estimated states of the real world.
            Idx (dict): A dictionary containing index mappings for variothe state and control signals.
        Returns:
            np.ndarray: A flattened array of actuator commands.
        """
        if self.bypass_controller:
            return np.zeros((Idx["NU"],))
        
        self.est_world_states = est_world_states
       
        # feedforward and feedback controller
        mtb_torque_cmd, rw_torque_cmd = self.controller_algorithm.get_dipole_moment_and_rw_torque_command(est_world_states, Idx)
        
        # actuator management function
        actuator_cmd = self.allocate_torque(mtb_torque_cmd, rw_torque_cmd, Idx)

        # convert dipole moment to voltage command     
        # TODO: write a "convert to output format" function
        # possibilities for rw are volt, curr, speed or torque. If speed, integrator will be needed
        volt_cmd = np.zeros((Idx["NU"],))
        for i in range(len(self.mtb)):
            volt_cmd[Idx["U"]["MTB_TORQUE"]][i] = self.mtb[i].convert_dipole_moment_to_voltage(actuator_cmd[Idx["U"]["MTB_TORQUE"]][i])
        # [To determine:] RW interface. Baseline torque/current control
        volt_cmd[Idx["U"]["RW_TORQUE"]] = actuator_cmd[Idx["U"]["RW_TORQUE"]]

        return volt_cmd.flatten()

        