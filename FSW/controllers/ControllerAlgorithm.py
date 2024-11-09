import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *

class ControllerAlgorithm:
    def __init__(self, Magnetorquers: list, ReactionWheels: list, params: dict):
        self.G_mtb_b = np.array(params["mtb_orientation"]).reshape(params["N_mtb"], 3)
        self.G_rw_b  = np.array(params["rw_orientation"]).reshape(params["N_rw"], 3)
        self.J = np.array(params["inertia"]).reshape(3,3)
        max_moms = np.zeros(len(Magnetorquers))
        for i, mtb in enumerate(Magnetorquers):
            max_moms[i] = mtb.max_dipole_moment
        
        self.ubmtb = np.min(np.abs(self.G_mtb_b).T @ max_moms)
        self.lbmtb = -self.ubmtb

        self.ubrw = np.array([rw.max_torque for rw in ReactionWheels])
        self.lbrw = -self.ubrw
        self.G_rw_b  = np.array([rw.G_rw_b for rw in ReactionWheels]).reshape(3, -1)


    def get_dipole_moment_and_rw_torque_command(self, state: np.ndarray, Idx: dict):
        raise NotImplementedError("This method should be overridden by subclasses")
