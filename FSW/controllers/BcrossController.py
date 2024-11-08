import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *
from FSW.controllers.ControllerAlgorithm import ControllerAlgorithm  # Assuming this is the correct import path

class BcrossController(ControllerAlgorithm):
    def __init__(
        self,
        Magnetorquers: list, 
        ReactionWheels: list, 
        params: dict,
    ) -> None:
        super().__init__(Magnetorquers, ReactionWheels, params)  
        self.k = np.array(params["bcrossgain"])
        
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction
                
        self.ref_angular_velocity = self.I_min_direction * np.deg2rad(params["tgt_ss_ang_vel"])
 
    def get_dipole_moment_and_rw_torque_command(
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
        return np.clip(a=m_cmd, a_min=self.lbmtb, a_max=self.ubmtb), []
