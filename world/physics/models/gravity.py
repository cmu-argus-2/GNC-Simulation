# built-in imports
import numpy as np
import brahe
import brahe.orbit_dynamics
from brahe.epoch import Epoch

'''
    CLASS GRAVITY

    Defines the gravity acceleration of the satellite at any point in the simulation
'''
class Gravity():
    def __init__(self) -> None:
        pass


    '''
        FUNCTION ACCELERATION

        Computes the acceleration due to gravity on the satellite CoM

        INPUTS:
            1. r - satellite position in ECI frame [UNITS : m]
            2. epc - current Epoch as an instance of brahe's Epoch class
        
        OUTPUTS:
            1. a - acceleration due to gravity in ECI frame
    '''
    def acceleration(self, r:np.ndarray, epc:Epoch):

        R_ECI2ECEF = brahe.frames.rECItoECEF(epc)
        a = brahe.orbit_dynamics.gravity.accel_gravity(r, R_ECI2ECEF, 2,0) # (2,0) corresponds to J2 variation
        return a