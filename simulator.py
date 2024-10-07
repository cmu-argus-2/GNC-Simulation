# built-in imports
import yaml
import numpy as np

# file imports
from world.physics.dynamics import Dynamics

from FSW.controllers.controller import Controller
from FSW.estimators.estimator import Estimator
from visualization.plot_results import main_plotter

"""
    CLASS MANAGER
    Master class that controls the entire simulation
"""

class Simulator:
    def __init__(self) -> None:
        self.load_config()

        # Update rate: ideally this should just be the world update rate
        self.update_rate = np.max(
            [
                self.config["solver"]["world_update_rate"],
                self.config["solver"]["controller_update_rate"],
                self.config["solver"]["payload_update_rate"],
            ]
        )

        # Initialize the index dictionary
        self.Idx = {}
        
        # Intialize the dynamics class as the "world"
        self.world = Dynamics(self.config, self.Idx)
        self.Idx = self.world.Idx
        
        # Initialize the Controller Class
        self.controller = Controller(self.config, self.Idx)
        
        self.estimator = Estimator()
        
        # Initialize state and date
        self.world_state    = None
        self.measurements   = None
        self.control_inputs = None
        
        self.date = self.config["mission"]["start_date"]

    """
        FUNCTION LOAD_CONFIG
        Loads the config file as a dictionary into this class

        INPUTS:
            None
        
        OUTPUTS:
            None
    """

    def load_config(self):
        # Read the YAML file and parse it into a dictioanry
        self.config_file = "config.yaml"
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    """
        FUNCTION RUN
        Runs the simulation until a termination is reached

        INPUTS:
            None

        OUTPUTS:
            None
    """

    def run(self):
        
        Results         = {}
        Results["Time"] = []
        TimeVec         = np.arange( 0,self.config["mission"]["duration"], 1 / self.update_rate)
        NT              = len(TimeVec)
        Results["Time"] = TimeVec
        WorldStates     = np.zeros((NT, self.Idx["NX"]))
        Results["WorldStates"]   = WorldStates
        Results["ControlInputs"] = np.zeros((NT, self.Idx["NU"]))
        
        for i, _ in enumerate(TimeVec):
            self.world_state = self.world.state

            # self.measurement = self.sensors(self.date, self.world_state) # UNCOMMENT WHEN SENSOR MODELS ARE IMPLEMENTED
            self.measurement = []
            # self.control_inputs = self.controller(self.date, self.measurement)
            est_world_states = self.estimator.run(self.date, self.measurement, self.world_state, self.Idx)
            
            actuator_cmd = self.controller.run(self.date, est_world_states, self.Idx)

            # Pedro: may want to rename dynamics to world. It will include the propagation of 
            # actuator states (rw speed, motor time constant, ...)
            self.world.update(input=actuator_cmd) # self.control_torques)
            
            self.date += (1 / self.update_rate) / (
                24 * 60 * 60
            )  # 1 second into Julian date conversion
            print(f"{self.date:.2f}", np.round(self.world_state, 2))
            
            Results["WorldStates"][i, :] = self.world_state
            Results["ControlInputs"][i, :] = actuator_cmd[:, 0]
            # self.world.Magnetorquers.get_power 
            # self.world.ReactionWheels
            # Results["PowerConsumption"][i] = self.actuation.get_power_consumption()
        
        return Results


if __name__ == "__main__":
    simulator = Simulator()
    Results = simulator.run()
    
    save_results = True
    if save_results:
        import pickle
        with open("Results.pkl", "wb") as f:
            pickle.dump({"Results": Results, "Idx": simulator.Idx}, f)
            
    main_plotter(Results, simulator.Idx)
