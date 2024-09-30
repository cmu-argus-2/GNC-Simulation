import numpy as np
from actuators.magnetorquer import Magnetorquer
from world.math.quaternions import *


class SpinStabilizationController():
    def __init__(
        self,
        magnetorquer: Magnetorquer,
        b_dot_gain: float,
        ub_voltage: float,
        ub_current: float,
        lb_voltage=0.0,
        lb_current=0.0,
    ) -> None:
        self.model = magnetorquer
        self.k = b_dot_gain
        self.lbv = lb_voltage
        self.ubv = ub_voltage
        self.lbi = lb_current
        self.ubi = ub_current

    def get_voltage_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray
    ) -> np.ndarray:
        v_cmd = self.model.convert_dipole_moment_to_voltage(
            self.get_dipole_moment_command(magnetic_field, angular_velocity)
        )
        return np.clip(v_cmd, a_min=self.lbv, a_max=self.ubv)

    def get_current_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        i_cmd = self.model.convert_dipole_moment_to_current(
            self.get_dipole_moment_command(magnetic_field, angular_velocity)
        )
        return np.clip(i_cmd, a_min=self.lbi, a_max=self.ubi)

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        B-dot law: https://arc.aiaa.org/doi/epdf/10.2514/1.53074
        """
        magnetic_field_norm = np.linalg.norm(magnetic_field, ord=2)
        unit_magnetic_field = magnetic_field / magnetic_field_norm
        return -self.k / magnetic_field_norm \
                       * np.cross(unit_magnetic_field, angular_velocity)
