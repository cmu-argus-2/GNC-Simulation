import numpy as np


class LyapunovSunPointingController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        dipole_moment_lower_bound=np.finfo(np.float64).min,
        dipole_moment_upper_bound=np.finfo(np.float64).max,
    ) -> None:
        # [TODO] max moment should be defined per axis based on all magnetorquers, not just first
        self.J = inertia_tensor
        self.lbm = dipole_moment_lower_bound
        self.ubm = dipole_moment_upper_bound

    def get_dipole_moment_command(
        self,
        magnetic_field: np.ndarray,
        angular_velocity: np.ndarray,
        sc_inertia: np.ndarray,
        target_angular_velocity: np.ndarray,
        sun_vector: np.ndarray,
        max_dipole_moment: float,
    ) -> np.ndarray:
        """
        Lyapunov-based sun-pointing law:
            https://digitalcommons.usu.edu/smallsat/2024/all2024/56/
        """
        # omega sat body frame ang vel
        # omega_b gyro bias
        # I_max - direction of axis of greatest inertia

        h = sc_inertia @ angular_velocity
        h_norm = np.linalg.norm(h)
        h_tgt = sc_inertia @ target_angular_velocity
        h_tgt_norm = np.linalg.norm(h_tgt)
        I_max_index = np.argmax(np.diag(sc_inertia))
        I_max_direction = np.zeros(3)
        I_max_direction[I_max_index] = 1.0
        if np.linalg.norm(I_max_direction - h / h_tgt_norm) > 0.1:
            u = magnetic_field @ (I_max_direction - h/h_tgt_norm)
            u = max_dipole_moment * u/np.linalg.norm(u)
        elif np.linalg.norm(sun_vector-h/h_norm)>0.01:
            u = magnetic_field @ (sun_vector - h/h_norm)
            u = max_dipole_moment * u/np.linalg.norm(u)
        else:
            u = np.zeros(3)
        return u

