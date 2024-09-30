import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *


class DetumblingController():
    def __init__(
        self,
        b_dot_gain: float,
        dipole_moment_lower_bound=np.zeros(3),
        dipole_moment_upper_bound=np.finfo(np.float64).max,
    ) -> None:
        self.k = b_dot_gain
        self.lbm = dipole_moment_lower_bound
        self.ubm = dipole_moment_upper_bound

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        B-cross law: https://arc.aiaa.org/doi/epdf/10.2514/1.53074
        """
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        unit_magnetic_field = magnetic_field / magnetic_field_norm
        m_cmd = -self.k / magnetic_field_norm \
                        * np.cross(unit_magnetic_field, angular_velocity)
        return np.clip(a=m_cmd, a_min=self.lbm, a_max=self.ubm)
