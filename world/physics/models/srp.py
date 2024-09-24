# built-in imports
import brahe.orbit_dynamics
import brahe.orbit_dynamics.srp
import numpy as np
import brahe
from brahe.epoch import Epoch

# file imports
from world.math.quaternions import quatrotation

'''
    CLASS SRP

    Calculates acceleration vector of the satellite due to solar radiation pressure
'''
class SRP():
    def __init__(self) -> None:
        pass

    '''
        FUNCTION ACCLERATION

        computes the solar radiation pressure induced acceleration of the satellite in orbit

        INPUTS:
            1. r - position vector of the satellite in ECI frame [UNITS : m]
            2. q - quaternion representing satellite body frame to ECI rotation
            2. epc - current Epoch as an instance of brahe's Epoch class
            3. satellite - dictionary containing mass and area properties of the satellite

        OUTPUTS:
            1. a - srp acceleration of the satellite in ECI frame (m/s^2)

        TODO: Adjust satellite's illumination area based on attitude relative to satellite-sun vector
    '''
    def acceleration(self, r:np.ndarray, q:np.ndarray, epc:Epoch, satellite:dict):
        
        r_sun =  self.sun_position(epc) # sun position in ECI frame

        illumination_factor = brahe.orbit_dynamics.srp.eclipse_conical(r, r_sun) # 1 if fully illuminated and 0 if completely in eclipse

        area = self.frontal_area(q, r - r_sun)*satellite["area"] # EDIT after including attitude as a parameter

        a = illumination_factor*brahe.orbit_dynamics.srp.accel_srp(r, r_sun, satellite["mass"], area)
        
        return a
    
    '''
        FUNCTION FRONTAL_AREA_FACTOR
        Computes the frontal area in the direction of the velocity vector

        INPUTS:
            1. q - quaternion attitude from Body frame to ECI
            2. r_bs - relative position vector between sun and body in ECI
        
        OUTPUTS
            1. k - frontal area factor for solar illumination calculations
    '''
    def frontal_area(self, q:np.ndarray, r_bs:np.ndarray):
        R_BtoECI = quatrotation(q)

        k = (np.abs(np.dot(R_BtoECI[:,0], r_bs)) + np.abs(np.dot(R_BtoECI[:,1], r_bs)) + np.abs(np.dot(R_BtoECI[:,2], r_bs)))/np.linalg.norm(r_bs)
        return k
    
    '''
        FUNCTION SUN_POSITION

        Computes the Sun position in ECI frame at a given time

        INPUTS:
            1. epc - epoch as an instance of brahe's Epoch class
        
        OUTPUTS:
            1. r_sun - position vector of the sun in ECI frame [UNITS: m]
    '''
    def sun_position(self, epc:Epoch):
        r_sun = brahe.ephemerides.sun_position(epc)

        return r_sun

        