# built-in imports
import yaml
import numpy as np

# file imports
from world.physics.dynamics import Dynamics

'''
    CLASS MANAGER
    Master class that controls the entire simulation
'''
class Simulator():
    def __init__(self) -> None:
        self.load_config()

        # Update rate: ideally this should just be the world update rate
        self.update_rate = np.max([self.config["solver"]["world_update_rate"], self.config["solver"]["controller_update_rate"], self.config["solver"]["payload_update_rate"]])

        # Intialize the dynamics class as the "world"
        self.world = Dynamics(self.config)

        # Initialize state and date
        self.world_state = None
        self.measurements = None
        self.control_inputs = None

        self.date = self.config["mission"]["start_date"]

    '''
        FUNCTION LOAD_CONFIG
        Loads the config file as a dictionary into this class

        INPUTS:
            None
        
        OUTPUTS:
            None
    '''
    def load_config(self):
        # Read the YAML file and parse it into a dictioanry
        self.config_file = 'config.yaml'
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    
    '''
        FUNCTION RUN
        Runs the simulation until a termination is reached

        INPUTS:
            None

        OUTPUTS:
            None
    '''
    def run(self):
        while self.date - self.config["mission"]["start_date"] <= self.config["mission"]["duration"]/(24*60*60):
            self.world_state = self.world.state

            # self.measurement = self.sensors(self.date, self.world_state) # UNCOMMENT WHEN SENSOR MODELS ARE IMPLEMENTED
            # self.control_inputs = self.controller(self.date, self.measurement) 
            # self.control_torques = self.actuators(self.date, self.measurement)

            self.world.update(input=np.zeros((9,)))

            self.date += (1/self.update_rate)/(24*60*60) # 1 second into Julian date conversion
            print(self.date, self.world_state)


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()

    
