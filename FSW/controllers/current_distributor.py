from typing import List
import numpy as np
from actuators.magnetorquer import Magnetorquer

class MagnetorquerCurrentDistributor():
    def __init__(
        self,
        magnetorquers: List[Magnetorquer]
    ) -> None:
        self.D = self.get_current_distribution_matrix(magnetorquers)

    def get_current_commands(
        self,
        dipole_moment: np.ndarray,
    ) -> np.ndarray:
        return self.D @ dipole_moment

    def get_current_distribution_matrix(
        self,
        magnetorquers: List[Magnetorquer]
    ) -> np.ndarray:
        U = magnetorquers[0].G_mtb_b
        k = np.array([magnetorquers[0].get_dipole_moment_over_current()])

        for i in range(1, len(magnetorquers)):
            U = np.column_stack((U, magnetorquers[i].G_mtb_b))
            k = np.hstack((
                k, magnetorquers[i].get_dipole_moment_over_current()
            ))

        K = np.diag(k)
        return np.linalg.pinv(U @ K)
