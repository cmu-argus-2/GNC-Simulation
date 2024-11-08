
import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *
from FSW.controllers.ControllerAlgorithm import ControllerAlgorithm  


class BaselineSunPointingController(ControllerAlgorithm):
    def __init__(
        self,
        Magnetorquers: list,
        ReactionWheels: list,
        params: dict,
    ) -> None:
        # [TODO] max moment should be defined per axis based on all magnetorquers, not just first
        super().__init__(Magnetorquers, ReactionWheels, params)

        self.kdetumb = np.array(params["bcrossgain"])
        self.k = np.array(params["mtb_att_feedback_gains"])
        
        eigenvalues, eigenvectors = np.linalg.eig(self.J)
        I_min_direction = eigenvectors[:, np.argmax(eigenvalues)]
        if I_min_direction[np.argmax(np.abs(I_min_direction))] < 0:
            I_min_direction = -I_min_direction
        self.I_min_direction = I_min_direction 
        self.target_angular_velocity = np.array(params["tgt_ss_ang_vel"])
        self.ref_angular_velocity = self.I_min_direction * np.deg2rad(self.target_angular_velocity)
        self.h_tgt = self.J @ self.ref_angular_velocity
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
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
 
        unit_magnetic_field = magnetic_field / (magnetic_field_norm ** 2)

        if not spin_stabilized:
            u = self.kdetumb * np.cross(unit_magnetic_field, self.ref_angular_velocity - angular_velocity).reshape(3,1)
        elif not sun_pointing:
            Bhat = crossproduct(magnetic_field)
            Bhat_pseudo_inv = Bhat.T / (magnetic_field_norm ** 2)

            err_vec = np.hstack((-self.I_min_direction-sun_vector, self.ref_angular_velocity - angular_velocity))
            u = Bhat_pseudo_inv @ self.k @ err_vec

        u = np.clip(a=u, a_min=self.lbmtb, a_max=self.ubmtb)

        return u.reshape(3, 1), []
