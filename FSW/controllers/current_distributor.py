import numpy as np
from actuators.magnetorquer import Magnetorquer

class TripleCurrentDistributor():
    def __init__(
        self,
        torquer1: Magnetorquer,
        torquer2: Magnetorquer,
        torquer3: Magnetorquer,
    ) -> None:
        self.D = self.get_current_distributor_matrix(
            torquer1, torquer2, torquer3
        )

    def get_current_command(
        self,
        dipole_moment: np.ndarray,
    ) -> np.ndarray:
        return self.D @ dipole_moment

    def get_current_distributor_matrix(
        self,
        torquer1: Magnetorquer,
        torquer2: Magnetorquer,
        torquer3: Magnetorquer,
    ) -> np.ndarray:
        U = np.column_stack((
            torquer1.G_mtb_b, torquer2.G_mtb_b, torquer3.G_mtb_b
        ))
        K = np.diag(np.array([
            torquer1.get_dipole_moment_over_current(),
            torquer2.get_dipole_moment_over_current(),
            torquer3.get_dipole_moment_over_current(),
        ]))
        return np.linalg.pinv(U @ K)
