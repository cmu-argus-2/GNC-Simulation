# built-in imports
import numpy as np
import brahe
from brahe.epoch import Epoch
import pyIGRF

"""
    CLASS MAGNETICFIELD

    Uses the IGRF model to compute the magnetic field vector any any given position
"""


class MagneticField:
    def __init__(self) -> None:
        pass

    """
        FUNCTION FIELD
        Computes the magnetic field vector in ECI frame given the satellite position and date

        INPUTS:
            1. r - satellite position in ECI frame [UNITS: m]
            2. epc - current Epoch as an instance of brahe's Epoch

        OUTPUTS:
            1. B - mangetic field vector in ECI frame [UNITS: T]
    """

    def field(self, r: np.ndarray, epc: Epoch):
        R_ECI2ECEF = brahe.frames.rECItoECEF(epc)
        r_ecef = R_ECI2ECEF @ r
        longitude, latitude, altitude = brahe.coordinates.sECEFtoGEOC(r_ecef, True)

        _, _, _, BN, BE, BD, _ = pyIGRF.igrf_value(
            latitude, longitude, altitude / 1.0e3, epc.year()
        )

        B_enz = np.array(
            [BE, BN, -BD]
        )  # magnetic field vector in ENZ frame [UNITS: nT]
        B_ecef = brahe.coordinates.sENZtoECEF(r_ecef, B_enz)
        B = R_ECI2ECEF.T @ B_ecef

        # Convert Units from nT to T
        return B * 1.0e-9
