import numpy as np


class LyapunovSunPointingController():
    def __init__(
        self,
        inertia_tensor: np.ndarray,
        mtb_config: dict,
    ) -> None:
        # [TODO] max moment should be defined per axis based on all magnetorquers, not just first
        self.J = inertia_tensor
        max_voltage = mtb_config["max_voltage"][0]
        max_power = mtb_config["max_power"][0]
        max_current_rating = mtb_config["max_current_rating"][0]
        max_current = np.min([max_power / max_voltage, max_current_rating])
        trace_thickness = mtb_config["trace_thickness"][0]
        N = mtb_config["coils_per_layer"][0]
        trace_width = mtb_config["trace_width"][0]
        gap_width = mtb_config["gap_width"][0]
        coil_width = trace_width + gap_width
        pcb_layers = mtb_config["layers"][0]
        N_per_face = N * pcb_layers
        pcb_side_max = 0.1
        A_cross = (pcb_side_max - N * coil_width) ** 2
        coil_length = 4 * (pcb_side_max - N*coil_width) \
                        * N * pcb_layers
        COPPER_RESISTIVITY = 1.724 * 10**-8

        R =  COPPER_RESISTIVITY * coil_length \
            / (trace_width * trace_thickness)

        self.ubm = N_per_face * max_current * A_cross

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

