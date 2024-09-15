# built-in imports
import numpy as np

class SolarRadiationPressure():
    def __init__(self, sim_params) -> None:
        self.sim_params = sim_params
        
    
    def acceleration(self):
        return np.array([0,0,0])