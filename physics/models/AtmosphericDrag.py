#built-in imports
import numpy as np


class AtmosphericDrag():
    def __init__(self, sim_params) -> None:
        self.sim_params = sim_params
        
        if hasattr(self.sim_params, 'harris_priester_data'):
            self.hp_data = np.genfromtxt(self.sim_params.harris_priester_data, delimiter=',') # 3 columns [altitude (km), min density (g/km^3), max density (g/km^3)]
        else:
            raise Exception("Atmospheric Data not found")
    
    '''
        Performs all internal compoutations and returns acceleration due to gravitational effects
        
        INPUTS:
            1. r - satellite position in ECI frame
            2. v - satellite velocity in ECI frame
            
        OUTPUTS:
            1. a - acceleration vector due to all modeled drag effects
    '''    
    def acceleration(self, r, v):
        return self.drag_acceleration(v, r) 
    
    
    # INTERNAL METHODS - ENSURE THESE ARE NOT ACCESSED OUTSIDE THIS CLASS
    
    '''
        Computes Atmospheric Drag given the velocity vector
        NOTE: Assumes that the drag vector passes through the CoM and that the frontal area remains fixed
              This will need to change if we are to consider attitude effects on the drag vector
              
        NOTE 2: Since this model assumes ECI, the Earth's rotation is ignored i.e., the atmosphere is not stationary, but rotating
                This effect is not yet accounted for
              
        INPUTS:
            1. v - velocity of the satellite (m/s) in ECI frame
            2. r - position vector in ECI frame (m)
            
        OUTPUTS:
            1. a - drag acceleration in ECI frame-
    '''    
    def drag_acceleration(self, v, r):
        v = np.array(v)
        a = -0.5*self.density(r)*np.linalg.norm(v)*v/self.sim_params.ballistic_coefficient
        
        return a
        
    
    '''
        Computes the Atmospheric density given the satellite position vector using the Harris-Priester model
        NOTE: Ignores solar flux changes that would require date and time as inputs. 
              Instead the density is computed as the average of the min and max densities
              
        INPUTS:
            1. Position vector in ECI frame (m)
            
        OUTPUTS:
            1. Density in 
    '''
    def density(self, r):
        
        satellite_altitude = (np.linalg.norm(r) - self.sim_params.R_earth)/1000 # in km
        h_i = np.where(self.hp_data[:,0] >= satellite_altitude)[0][0] - 1 # can throw an error if altitude is beyond data range
        h_ip1 = h_i + 1
        
        H_m = (self.hp_data[h_i,0] - self.hp_data[h_ip1,0])/(np.log(self.hp_data[h_ip1,1]/self.hp_data[h_i,1]))
        H_M = (self.hp_data[h_i,0] - self.hp_data[h_ip1,0])/(np.log(self.hp_data[h_ip1,2]/self.hp_data[h_i,2]))
        
        rho_m = self.hp_data[h_i,1]*np.exp((self.hp_data[h_i,0] - satellite_altitude)/H_m)
        rho_M = self.hp_data[h_i,1]*np.exp((self.hp_data[h_i,0] - satellite_altitude)/H_M)
        
        return 0.5*(rho_m + rho_M)*1e-12 # convert g/km^3 to kg/m^3
        
        
        