# built-in imports
import numpy as np
import brahe
from brahe.epoch import Epoch

# file imports
from world.math.quaternions import quatrotation


'''
    CLASS DRAG

    Computes drag induced acceleration in orbit
    Accesses the Harris-Priester model for density data
'''
class Drag():
    def __init__(self, harris_priester_density_file) -> None:
        self.hp_data = np.genfromtxt(harris_priester_density_file, delimiter=',')

        # Earth angular rotation in ECEF frame
        self.OMEGA_VECTOR = np.array([0.0, 0.0, brahe.OMEGA_EARTH])

    '''
        FUNCTION ACCELERATION

        Computes the drag acceleration using the Harris-Priester density model
        NOTE : Drag does not seem to have a python version in brahe
               This file has been translated from the rust script (https://github.com/duncaneddy/brahe/blob/main/src/orbit_dynamics/drag.rs)
               Reference calculation : Satellite Orbits, Montenbruck & Gill Pg 89

        INPUTS:
            1. r - satellite position in ECI frame [UNITS : m]
            2. q - quaternion representing rotation from satellite body frame to ECI
            3. v - satellite velocity in ECI frame [UNITS: m/s]
            4. epc - current Epoch as an instance of brahe's Epoch class
            4. satellite - dictionary containing mass and area properties of the satellite

        OUTPUTS:
            1. a - drag-induced acceleration vector in ECI frame (m/s^2)
        
        TODO: Account for changing area due to attitude
    '''
    def acceleration(self, r:np.ndarray, v:np.ndarray, q:np.ndarray, epc:Epoch, satellite:dict):

        R_ECI2ECEF = brahe.frames.rECItoECEF(epc)
        r_ecef = R_ECI2ECEF@r
        v_ecef = R_ECI2ECEF@v
        
        # Adjust for Atmospheric moving with Earth
        v_rel = v_ecef - np.cross(self.OMEGA_VECTOR, r_ecef)
        
        # Area adjustment due to attitude
        area = self.frontal_area(q, v)*satellite["area"]

        # Get density
        density = self.harris_priester(r, R_ECI2ECEF, epc)

        # Acceleration in ECEF frame
        a = -0.5*satellite["Cd"]*area*density*np.linalg.norm(v_rel)*v_rel/satellite["mass"]

        return R_ECI2ECEF.T@a


    '''
        FUNCTION HARRIS_PRIESTER
        
        Calculates the atmospheric density at a given altitude and solar flux

        INPUTS:
            1. r - satellite position in ECI frame
            2. R - rotation matrix from ECI to ECEF frame
            2. epc - current Epoch as an instance of brahe's Epoch class

        OUTPUTS:
            1. density - atmospheric density at current satellite position [UNITS: kg/m^3]
    '''
    def harris_priester(self, r:np.ndarray, R: np.ndarray, epc:Epoch):
        HP_UPPER_LIMIT = 1000.0
        HP_LOWER_LIMIT = 100.0

        HP_RA_LAG = 0.523599 # pi/6
        HP_N_PRM = 3.0
        
        # Satellite height
        r_ecef = R@r
        r_geod = brahe.coordinates.sECEFtoGEOD(r_ecef, False)
        height = r_geod[2]/1.0e3

        # Exit with zero density outside height model limits
        if height >= HP_UPPER_LIMIT or height <= HP_LOWER_LIMIT:
            return 0.0
        
        # Sun right ascension, declination
        r_sun = brahe.ephemerides.sun_position(epc) # sun position in ECI frame
        r_sun_ecef = R@r_sun
        ra_sun = np.arctan2(r_sun_ecef[1]/r_sun_ecef[0])
        dec_sun = np.arctan2(r_sun_ecef[2]/(np.sqrt(r_sun_ecef[0]**2 + r_sun_ecef[1]**2)))

        # Unit vector u towards the apex of the diurnal bulge in inertial geocentric coordinates
        c_dec = np.cos(dec_sun)
        u = np.array([c_dec*np.cos(ra_sun + HP_RA_LAG), c_dec*np.sin(ra_sun + HP_RA_LAG), np.sin(dec_sun)])
        
        # Cosine of half angle between satellite position vector and apex of diurnal bulge
        c_psi2 = 0.5 + np.dot(r_ecef, u)/np.linalg.norm(r_ecef)

        # Exponential Desnity Interpolation
        ih = np.where(self.hp_data[:,0] >= height)[0][0] - 1
        
        h_min = (self.hp_data[ih,0] - self.hp_data[ih+1,0])/(np.log(self.hp_data[ih+1,1]/self.hp_data[ih,1]))
        h_max = (self.hp_data[ih,0] - self.hp_data[ih+1,0])/(np.log(self.hp_data[ih+1,2]/self.hp_data[ih,2]))
        
        d_min = self.hp_data[ih,1]*np.exp((self.hp_data[ih,0] - height)/h_min)
        d_max = self.hp_data[ih,1]*np.exp((self.hp_data[ih,0] - height)/h_max)

        density = d_min + (d_max-d_min)*c_psi2**HP_N_PRM

        # Convert from g/km^3 to kg/m^3
        return density*1.0e-12
    
    '''
        FUNCTION FRONTAL_AREA_FACTOR
        Computes the frontal area in the direction of the velocity vector

        INPUTS:
            1. q - quaternion attitude from Body frame to ECI
            2. v - velocity vector in ECI
        
        OUTPUTS
            1. k - frontal area factor for drag calculations
    '''
    def frontal_area(self, q:np.ndarray, v:np.ndarray):
        R_BtoECI = quatrotation(q)

        k = (np.abs(np.dot(R_BtoECI[:,0], v)) + np.abs(np.dot(R_BtoECI[:,1], v)) + np.abs(np.dot(R_BtoECI[:,2], v)))/np.linalg.norm(v)
        return k


        
