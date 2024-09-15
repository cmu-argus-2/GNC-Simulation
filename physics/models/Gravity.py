#built-in imports
import numpy as np

class Gravity():
    def __init__(self, sim_params) -> None:
        
        self.sim_params = sim_params
    
    '''
        Performs all internal compoutations and returns acceleration due to gravitational effects
        
        INPUTS:
            1. r - satellite position in ECI frame NOTE: in metres not km
            
        OUTPUTS:
            1. a - acceleration vector due to all modeled gravitational effects
    '''    
    def acceleration(self, r):
        return self.spherical_acceleration(r) , self.J2_perturbation(r)
    
    
    # INTERNAL METHODS - ENSURE THESE ARE NOT ACCESSED OUTSIDE THIS CLASS
    '''
        Computes the Gravitational acceleration vector assumping the Earth is perfectly spherical
        NOTE: Assumes the position vector 'r' is in ECI frame
        
        INPUTS:
            1. position vector 'r' in ECI frame
        
        OUTPUTS:
            1. acceleration vector 'a' in the same frame as 'r'
    '''
    def spherical_acceleration(self, r):
        r = np.array(r) # convert to a numpy array in case it wasn't already
        a = -(self.sim_params.mu/(np.linalg.norm(r))**3)*r
        return a
    
    '''
        Computes the J2 perturbation acceleration in the ECI frame
        NOTE: Assumes the position vector 'r' is in ECI frame
        
        Source : https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=3727&context=etd_theses
        
        INPUTS:
            1. position vector 'r' in ECI frame
            
        OUTPUTS:
            1. acceleration vector 'a' in the same frame as 'r'
    '''
    def J2_perturbation(self, r):
        
        r = np.array(r)
        rx = r[0]
        ry = r[1]
        rz = r[2]
        R = np.linalg.norm(r)
        
        a = (3*self.sim_params.J2*self.sim_params.mu*(self.sim_params.R_earth**2)/(2*R**5))*\
            np.array([(5*(rz**2/R**2) - 1)*rx, \
                      (5*(rz**2/R**2) - 1)*ry, \
                      (5*(rz**2/R**2) - 3)*rz])
            
        return a