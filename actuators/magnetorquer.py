import numpy as np
from world.math.quaternions import crossproduct, quatrotation


COPPER_RESISTIVITY = 1.724 * 10**-8


# This is a single torquer
class Magnetorquer:
    def __init__(
        self,
        mtb_config: dict,
        mtb_id: int,
    ) -> None:
        self.max_voltage = mtb_config["max_voltage"][mtb_id]
        self.N = mtb_config["coils_per_layer"][mtb_id]
        self.pcb_layers = mtb_config["layers"][mtb_id]
        self.N_per_face = self.N * self.pcb_layers
        self.trace_thickness = mtb_config["trace_thickness"][mtb_id]
        self.trace_width = mtb_config["trace_width"][mtb_id]
        self.gap_width = mtb_config["gap_width"][mtb_id]
        self.coil_width =  self.trace_width + self.gap_width
        self.max_power = mtb_config["max_power"][mtb_id]
        self.max_current_rating = mtb_config["max_current_rating"][mtb_id]
        self.max_current = np.min([
            self.max_power / self.max_voltage, self.max_current_rating])
        # I_max = min Imax, Vmax / R
        # D_max = N * I_max * A * G

        self.pcb_side_max = 0.1
        self.A_cross = (self.pcb_side_max - self.N * self.coil_width) ** 2
        self.R = self.compute_coil_resistance()

        self.current = 0.0
        self.voltage = 0.0
        self.power   = 0.0
        self.G_mtb_b = np.array(mtb_config["mtb_orientation"][mtb_id]).T
        self.dipole_moment = np.zeros(3,)
        self.max_dipole_moment = self.current_to_dipole_moment(self.max_current)

    def get_torque(self, MAG_FIELD):
        """
        Update voltage or current before getting the torque.
        """
        # TODO: Get moment and field in whatever frame the sim wants torque
        return np.crossproduct(
            self.dipole_moment, MAG_FIELD
        )

    def get_power(self):
        return self.R * self.current ** 2

    def set_voltage(
        self,
        voltage: float,
    ) -> None:
        if voltage > self.max_voltage:
            raise ValueError("Voltage exceeds maximum voltage rating.")
        # Current driver is PWM
        self.voltage = voltage
        self.current = voltage / self.R
        self.power   = self.R * self.current ** 2
        if self.current > self.max_current:
            raise ValueError(
                f"Current exceeds maximum power limit of {self.max_power} W."
            )
        self.dipole_moment = self.current_to_dipole_moment(self.current)

    def set_current(
        self,
        current: float,
    ) -> None:
        if current > self.max_current:
            raise ValueError(
                f"Current exceeds maximum power limit of {self.max_power} W."
            )
        self.dipole_moment = self.current_to_dipole_moment(current)

    def get_dipole_moment_over_current(self) -> float:
        return self.N_per_face * self.A_cross

    def current_to_dipole_moment(
        self,
        current: float,
    ) -> np.ndarray:
        #TODO: confirm that G_mtb_b is unit vector
        return self.N_per_face * current * self.A_cross * self.G_mtb_b

    def dipole_moment_to_voltage(
        self,
        dipole_moment: np.ndarray,
    ) -> float:
        self.dipole_moment = dipole_moment
        I = self.dipole_moment_to_current(dipole_moment)
        self.current = np.clip(I, -self.max_current, self.max_current)
        # clip voltage to max voltage
        self.voltage = np.clip(
            self.current * self.R, -self.max_voltage, self.max_voltage)
        self.power   = self.R * self.current ** 2
        return self.voltage

    def voltage_to_dipole_moment(
        self,
        voltage: float,
    ) -> np.ndarray:
        self.voltage = voltage
        I = voltage / self.R
        # clip current to max current
        self.current = np.clip(I, -self.max_current, self.max_current)
        self.dipole_moment = self.current_to_dipole_moment(self.current)
        self.power   = self.R * self.current ** 2
        return self.dipole_moment

    def dipole_moment_to_current(
        self,
        dipole_moment: np.ndarray,
    ) -> float:
        return np.linalg.norm(dipole_moment) / self.N_per_face / self.A_cross

    def compute_coil_resistance(self):
        coil_length = 4 * (self.pcb_side_max - self.N*self.coil_width) \
                        * self.N * self.pcb_layers
        R =  COPPER_RESISTIVITY * coil_length \
            / (self.trace_width * self.trace_thickness)
        return R
