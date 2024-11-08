import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *
from FSW.controllers.ControllerAlgorithm import ControllerAlgorithm  

class LyapBasedSunPointingController(ControllerAlgorithm):
    def __init__(
        self,
        Magnetorquers: list, 
        ReactionWheels: list, 
        params: dict,
    ):
        super().__init__(Magnetorquers, ReactionWheels, params)  # Call the parent class's __init__ method

        self.k = np.array(params["bcrossgain"])
        
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction 
        self.h_tgt = self.J @ self.I_min_direction * np.deg2rad(params["tgt_ss_ang_vel"])
        self.h_tgt_norm = np.linalg.norm(self.h_tgt)

    def get_dipole_moment_and_rw_torque_command(
        self,
        est_ctrl_states: np.ndarray,
        Idx: dict,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        Re2b = quatrotation(est_ctrl_states[Idx["X"]["QUAT"]]).T
        magnetic_field   = Re2b @ est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        sun_vector       = Re2b @ est_ctrl_states[Idx["X"]["SUN_POS"]]
        sun_vector       = sun_vector / np.linalg.norm(sun_vector)

        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        u = np.zeros(3)
        spin_stabilized = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))
        sun_pointing = (np.linalg.norm(sun_vector-(h/h_norm))<= np.deg2rad(10))

        if not spin_stabilized:
            u = crossproduct(magnetic_field) @ (self.I_min_direction - (h/self.h_tgt_norm))
        elif not sun_pointing:
            u = crossproduct(magnetic_field) @ (sun_vector - (h/self.h_tgt_norm)) # (h/self.h_tgt_norm))
        
        if np.linalg.norm(u) == 0:
            u = np.zeros(3)
        else:
            u = self.ubmtb * u/np.linalg.norm(u) 
        
        return u.reshape(3, 1), []
