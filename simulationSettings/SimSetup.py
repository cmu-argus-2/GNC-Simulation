# built-in imports
import yaml


'''
    CLASS SIMSETUP
    Defines an instance that stores all simulation related parameters in the current run
    This class can be used to either read a yaml file for a standalone simulation or
    to have its objects initialized and updated in a loop for Monte-carlo analysis
'''
class SimSetup():
    def __init__(self, sim_params_file) -> None:
        
        # Read the YAML file and parse it into a dictioanry
        self.sim_params_file = sim_params_file
        with open(self.sim_params_file, 'r') as f:
            self.params = yaml.safe_load(f)
        
        # Set each key in the dictionary as a class sttribute
        # Allows us to do SimSetup.Mass instead of SimSetup.params['Mass]   
        for (k,v) in self.params.items():
            setattr(self, k, v)
            
            
    def get(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None
        
    def set(self, attr, val):
        setattr(self, attr, val)
            