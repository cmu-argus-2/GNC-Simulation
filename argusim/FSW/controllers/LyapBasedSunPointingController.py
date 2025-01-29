import numpy as np
from argusim.actuators import Magnetorquer
from argusim.world.math.quaternions import *
from argusim.FSW.controllers.ControllerAlgorithm import ControllerAlgorithm  

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
        self.sun_pointed = False

    def get_dipole_moment_and_rw_torque_command(
        
        self,
        est_ctrl_states: np.ndarray,
        Idx: dict,
    ) -> np.ndarray:
        """
        Calculate the dipole moment and reaction wheel torque command for the satellite's control system.
        This method implements a Lyapunov-based sun-pointing control law to determine the necessary control inputs
        for the satellite to achieve and maintain a desired orientation relative to the sun.
        Args:
            est_ctrl_states (np.ndarray): Estimated control states of the satellite, including quaternion, magnetic field,
                          angular velocity, and sun position.
            Idx (dict): Dictionary containing indices for accessing specific elements within the `est_ctrl_states` array.
        Returns:
            np.ndarray: A 3x1 array representing the dipole moment command.
            list: An empty list (reserved for future use or additional outputs).
        Control Logic:
            - The method first calculates the rotation matrix from the estimated quaternion.
            - It then computes the magnetic field, angular velocity, and normalized sun vector in the body frame.
            - The angular momentum `h` and its norm `h_norm` are calculated.
            - The control input `u` is initialized to zero.
            - The method checks if the satellite is spin-stabilized and sun-pointing based on predefined thresholds.
            - If the satellite is not spin-stabilized, the control input `u` is calculated to align the angular momentum
              with the direction of minimum inertia.
            - If the satellite is spin-stabilized but not sun-pointing, the control input `u` is calculated to align the
              angular momentum with the sun vector.
            - The control input `u` is normalized and scaled by `self.ubmtb` if it is non-zero.
        Note:
            - The method assumes that the input `est_ctrl_states` and `Idx` are correctly formatted and contain all
              necessary information.
            - The control law is based on the reference: https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia
        
        magnetic_field   = est_ctrl_states[Idx["X"]["MAG_FIELD"]]
        angular_velocity = est_ctrl_states[Idx["X"]["ANG_VEL"]]
        sun_vector       = est_ctrl_states[Idx["X"]["SUN_POS"]]

        h = self.J @ angular_velocity 
        h_norm = np.linalg.norm(h)
        u = np.zeros(3)
        
        spin_stabilized = (np.linalg.norm(self.I_min_direction - (h/self.h_tgt_norm)) <= np.deg2rad(15))
        fine_sun_pointing   = (np.linalg.norm(sun_vector-(h/self.h_tgt_norm))<= np.deg2rad(10))
        """
        coarse_sun_pointing = (np.linalg.norm(sun_vector-(h/self.h_tgt_norm))<= np.deg2rad(15))
        if not coarse_sun_pointing or not spin_stabilized:  
            self.sun_pointed = False
            sun_pointing = fine_sun_pointing
        if self.sun_pointed:
            sun_pointing = coarse_sun_pointing
        else:
            sun_pointing = fine_sun_pointing
        """
        if not spin_stabilized:
            # u = crossproduct(magnetic_field) @ (self.I_min_direction - (h/self.h_tgt_norm))
            u = crossproduct(magnetic_field) @ (self.I_min_direction*self.h_tgt_norm - h) * self.k

            if np.linalg.norm(u) == 0:
                u = np.zeros(3)
            else:
                u = self.ubmtb * np.tanh(u) 
        
        elif not fine_sun_pointing:
            u = crossproduct(magnetic_field) @ (sun_vector - (h/self.h_tgt_norm)) # (h/self.h_tgt_norm))
            if np.linalg.norm(u) == 0:
                u = np.zeros(3)
            else:
                u = self.ubmtb * u/np.linalg.norm(u) 
                # element-wise division
                # u = self.ubmtb * np.sign(u) 
        # else:
        #     self.sun_pointed = True
        
        return u.reshape(3, 1), []
