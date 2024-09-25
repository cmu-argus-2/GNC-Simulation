import numpy as np
from world.math.quaternions import crossproduct, quatrotation

class Magnetorquers:
    """
    Represents all 3 magnetorquers on the satellite.
    TODO: calculate induced dipole moment based on applied currents and magnetorquer properties.
    """

    def __init__(self, config):
        self.max_induced_dipole_moment = 1  # TODO: placeholder value
        self.G_mtb_b                   = np.array(config["satellite"]["mtb_orientation"]).T


    def get_applied_torque(self, induced_dipole_moment, state, Idx):
        """
        Calculates the torque applied by the magnetorquers given
        the induced dipole moment and the external magnetic field.

        The induced dipole moment is not the direct control input,
        it will eventually be calculated based on the applied current and magnetorquer properties.

        Given 3 orthogonal magnetorquers we can produce a dipole moment
        in any direction up to a maximum magnitude.

        :param induced_dipole_moment: The induced dipole moment of the satellite, in the body frame.
        :param B_external: The external magnetic field due to the Earth, in the body frame.
        :return: The torque applied by the magnetorquers, in the body frame.
        """
        m_norm = np.linalg.norm(induced_dipole_moment)
        if m_norm > self.max_induced_dipole_moment:
            induced_dipole_moment *= self.max_induced_dipole_moment / m_norm
            
        # Body Frame torque 
        return crossproduct(state[Idx["X"]["MAG_FIELD"]])  @ self.G_mtb_b @ induced_dipole_moment
