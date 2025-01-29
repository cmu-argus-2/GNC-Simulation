import numpy as np
from argusim.actuators import Magnetorquer
from argusim.world.math.quaternions import *

class ControllerAlgorithm:
    def __init__(self, Magnetorquers: list, ReactionWheels: list, params: dict):
        self.sample_time = params["controller_dt"]
        self.G_mtb_b = np.array(params["magnetorquers"]["mtb_orientation"]).reshape(params["magnetorquers"]["N_mtb"], 3)
        self.G_rw_b  = np.array(params["reaction_wheels"]["rw_orientation"]).reshape(params["reaction_wheels"]["N_rw"], 3)
        self.J = np.array(params["inertia"]["nominal_inertia"]).reshape(3,3)
        self.ubmtbi = np.array([mtb.max_dipole_moment for mtb in Magnetorquers])
        self.lbmtbi = -self.ubmtbi
        
        self.ubmtb = np.min(np.abs(self.G_mtb_b).T @ self.ubmtbi)
        self.lbmtb = -self.ubmtb

        self.ubtrw = np.array([rw.max_torque for rw in ReactionWheels])
        self.lbtrw = -self.ubtrw
        self.ubwrw = np.array([rw.max_speed for rw in ReactionWheels])
        self.lbwrw = np.array([-rw.max_speed for rw in ReactionWheels])
        self.G_rw_b  = np.array([rw.G_rw_b for rw in ReactionWheels]).reshape(3, -1)
        self.J_rw   = np.array([rw.I_rw for rw in ReactionWheels])


    def get_dipole_moment_and_rw_torque_command(self, state: np.ndarray, Idx: dict):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def clip_total_dipole_moment(self, dipole_moment: np.ndarray) -> np.ndarray:
        """
        Clips the total dipole moment generated by all magnetorquers to be within the specified bounds.
        Parameters:
        dipole_moment (np.ndarray): The array representing the dipole moments to be clipped.
        Returns:
        np.ndarray: The clipped dipole moments within the lower and upper bounds.
        """

        return np.clip(dipole_moment, self.lbmtb, self.ubmtb)
    
    def clip_ind_dipole_moment(self, ind_dipole_moment: np.ndarray) -> np.ndarray:
        """
        Clips the dipole moment for each magnetorquer individually.
        Parameters:
        ind_dipole_moment (np.ndarray): The individual dipole moments to be clipped.
        Returns:
        np.ndarray: The clipped dipole moments within the specified bounds.
        """

        return np.clip(ind_dipole_moment, self.lbmtbi, self.ubmtbi)

    def clip_rw_torque(self, rw_torque: np.ndarray, current_angular_velocity: np.ndarray) -> np.ndarray:
        # Clip the torque to the maximum allowable torque
        clipped_torque = np.clip(rw_torque, self.lbtrw, self.ubtrw)
        
        # Calculate the new angular velocity
        new_angular_velocity = current_angular_velocity + (clipped_torque / self.J_rw) * self.sample_time
        
        # Clip the new angular velocity to the maximum allowable angular velocity
        clipped_angular_velocity = np.clip(new_angular_velocity, self.lbwrw, self.ubwrw)
        
        # Recalculate the torque based on the clipped angular velocity
        final_torque = (clipped_angular_velocity - current_angular_velocity) * self.J_rw / self.sample_time
        
        return final_torque