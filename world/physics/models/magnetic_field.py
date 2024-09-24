# built-in imports
import numpy as np
import brahe
from brahe.epoch import Epoch
import pyIGRF

'''
    CLASS MAGNETICFIELD

    Uses the IGRF model to compute the magnetic field vector any any given position
'''
class MagneticField():
    def __init__(self) -> None:
        pass

    '''
        FUNCTION FIELD
        Computes the magnetic field vector in ECI frame given the satellite position and date

        INPUTS:
            1. r - satellite position in ECI frame [UNITS: m]
            2. epc - current Epoch as an instance of brahe's Epoch

        OUTPUTS:
            1. B - mangetic field vector in ECI frame [UNITS: T]
    '''
    def field(self, r:np.ndarray, epc:Epoch):
        
        R_ECI2ECEF = brahe.frames.rECItoECEF(epc)
        r_ecef = R_ECI2ECEF@r
        longitude, latitude, altitude = brahe.coordinates.sECEFtoGEOC(r_ecef, True)

        _,_,_, BN, BE, BD, _ = pyIGRF.igrf_value(latitude, longitude, altitude/1.0e3, epc.year())

        B_enz = np.array([BE, BN, -BD]) # magnetic field vector in ENZ frame [UNITS: nT]
        B_ecef = brahe.coordinates.sENZtoECEF(r_ecef, B_enz)
        B = R_ECI2ECEF.T@B_ecef

        # Convert Units from nT to T
        return B*1.0e-9
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNIT TESTS

if __name__ == "__main__":
    magnetic_field = MagneticField()
    epc = Epoch(2024, 9, 23, 12, 0, 0, 0)

    # 1.
    r = np.array([6380000, 0, 0])
    r_geoc = brahe.coordinates.sECEFtoGEOC(brahe.frames.sECItoECEF(epc, r), use_degrees=True)
    print(r_geoc)
    acc = magnetic_field.field(r, epc) # should be 9.81 m/s^2
    print(epc.mjd(), acc, np.linalg.norm(acc))

    # 2.
    r = np.array([6500000/np.sqrt(2), 6500000/np.sqrt(2), 0])
    r_geoc = brahe.coordinates.sECEFtoGEOC(brahe.frames.sECItoECEF(epc, r), use_degrees=True)
    print(r_geoc)
    acc = magnetic_field.field(r, epc) # should not have J2 variations since it is in equatorial plane
    print(epc.mjd(), acc, np.linalg.norm(acc))

    # 3.
    r = np.array([6500000/np.sqrt(3), 6500000/np.sqrt(3), 6500000/np.sqrt(3)])
    r_geoc = brahe.coordinates.sECEFtoGEOC(brahe.frames.sECItoECEF(epc, r), use_degrees=True)
    print(r_geoc)
    acc = magnetic_field.field(r, epc) # should have J2 variation due to the Z axis
    print(epc.mjd(), acc, np.linalg.norm(acc))


