# Main entry point for each trial in a Python Job

from build.world.pyphysics import rk4
from build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from build.sensors.pysensors import readSensors
from simulation_manager import logger
from FSW.controllers.controller import Controller
from FSW.controllers.InitController import initialize_controller
from actuators.magnetorquer import Magnetorquer
from actuators.reaction_wheels import ReactionWheel

from time import time
import numpy as np
import os
import yaml

class Simulator():
    def __init__(self, trial_number, log_directory, config_path) -> None:
        
        self.trial_number = trial_number

        # Datapaths
        self.config_path = config_path
        self.log_directory = log_directory
        
        # Spacecraft Config
        self.params = SimParams(self.config_path, self.trial_number, self.log_directory)
        self.num_RWs = self.params.num_RWs
        self.num_MTBs = self.params.num_MTBs
        self.num_photodiodes = self.params.num_photodiodes

        # Controller Config
        with open(config_path, "r") as f:
            self.controller_params = yaml.safe_load(f)
        self.define_controller()

        # Initialization
        self.state = np.array(self.params.initial_state)
        self.J2000_start_time = self.params.sim_start_time
        self.control_input = np.zeros((self.params.num_MTBs + self.params.num_RWs))

        # Logging
        self.logr = logger.MultiFileLogger(log_directory)
        self.state_labels = ["r_x ECI [m]", "r_y ECI [m]", "r_z ECI [m]", "v_x ECI [m/s]", "v_y ECI [m/s]", "v_z ECI [m/s]",
                             "q_w", "q_x", "q_y", "q_z", "omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]", 
                             "rSun_x ECI [m]","rSun_y ECI [m]","rSun_z ECI [m]","xMag ECI [T]","yMag ECI [T]","zMag ECI [T]"] + \
                            ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.num_RWs)]
        
        self.measurement_labels = ["gps_posx ECEF [m]", "gps_posy ECEF [m]", "gps_posz ECEF [m]", "gps_velx ECEF [m/s]", "gps_vely ECEF [m/s]", "gps_velz ECEF [m/s]",
                                   "gyro_x [rad/s]", "gyro_y [rad/s]", "gyro_z [rad/s]", "mag_x_body [T]", "mag_y_body [T]", "mag_z_body [T]"] + \
                                  ['light_sensor_lux ' + str(i) for i in range(self.num_photodiodes)]
        
        self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] + ["T_RW_" + str(i) + " [Nm]" for i in range(self.num_RWs)]

    def define_controller(self):
        '''
            Defines the world for the controller objects 
        '''
        self.Idx = {}
        # Intialize the dynamics class as the "world"
        self.Idx["NX"] = 19
        self.Idx["X"]  = dict()
        self.Idx["X"]["ECI_POS"]   = slice(0, 3)
        self.Idx["X"]["ECI_VEL"]   = slice(3, 6)
        self.Idx["X"]["TRANS"]     = slice(0, 6)
        self.Idx["X"]["QUAT"]      = slice(6, 10)
        self.Idx["X"]["ANG_VEL"]   = slice(10, 13)
        self.Idx["X"]["ROT"]       = slice(6, 13)
        self.Idx["X"]["SUN_POS"]   = slice(13, 16)
        self.Idx["X"]["MAG_FIELD"] = slice(16, 19)

        # Actuator specific data
        # self.ReactionWheels = [ReactionWheel(self.config, IdRw) for IdRw in range(self.config["satellite"]["N_rw"])]

        # Actuator Indexing
        self.num_RWs  = self.controller_params["reaction_wheels"]["N_rw"]
        self.num_MTBs = self.controller_params["magnetorquers"]["N_mtb"]
        self.Idx["NU"]    = self.num_RWs + self.num_MTBs
        self.Idx["N_rw"]  = self.num_RWs
        self.Idx["N_mtb"] = self.num_MTBs
        self.Idx["U"]  = dict()
        self.Idx["U"]["MTB_TORQUE"]  = slice(0, self.num_MTBs)
        self.Idx["U"]["RW_TORQUE"] = slice(self.num_MTBs, self.num_RWs + self.num_MTBs)
        # RW speed should be a state because it depends on the torque applied and needs to be propagated
        self.Idx["NX"] = self.Idx["NX"] + self.num_RWs
        self.Idx["X"]["RW_SPEED"]   = slice(19, 19 + self.num_RWs)
        
        Magnetorquers = [Magnetorquer(self.controller_params["magnetorquers"], IdMtb) for IdMtb in range(self.num_MTBs)] 
        ReactionWheels = [ReactionWheel(self.controller_params["reaction_wheels"], IdRw) for IdRw in range(self.num_RWs)]
        self.controller = Controller(self.controller_params, Magnetorquers, ReactionWheels, self.Idx)

        # Controller Frequency
        self.controller_dt = self.controller_params['controller_dt']
        self.estimator_dt = self.controller_params['estimator_dt']

    def set_control_input(self, u):
        '''
            Sets the control input field of the class
            Exists for FSW to provide control inputs
        '''
        if len(u) < len(self.control_input) - 1:
            raise Exception("Control Input not provided to all Magnetorquers")
        elif len(u) == len(self.control_input)-1:
            self.control_input[0:len(u)] = u # Only magnetorquers
        else:
            self.control_input = u # magnetorquers + RWs
    
    def sensors(self, current_time, state):
        '''
            Implements partial observability using sensor models
        '''
        return readSensors(state, current_time, self.params)

    def step(self, sim_time, dt):
        '''
            Executes a single simulation step of a given step size
            This function is written separately to allow FSW to access simualtion stepping
        '''
        # Time
        current_time = self.J2000_start_time + sim_time
        
        # Get control input
        control_input = self.control_input

        # Step through the simulation
        self.state = rk4(self.state, control_input, self.params, current_time, dt)

        # Mask state through sensors
        measurement = self.sensors(current_time, self.state)
        
        # Log pertinent Quantities
        self.logr.log_v(
            "state_true.bin",
            [current_time] + self.state.tolist() + measurement.tolist() + control_input.tolist(),
            ["Time [s]"] + self.state_labels + self.measurement_labels + self.input_labels
        )

        return measurement
    
    def run(self):
        '''
            Runs the entire simulation
            Calls the 'step' function to run through the sim

            NOTE: This function operates on delta time i.e., seconds since mission start
                  This is done to allow FSW to provide a delta time to step() 
        '''

        WALL_START_TIME = time()

        # Load Sim start times
        sim_delta_time = 0
        last_controller_update = 0
        last_print_time = 0

        # Run iterations
        while sim_delta_time <= self.params.MAX_TIME:
            
            # Update the controller
            if sim_delta_time >= last_controller_update + self.controller_dt:            
                self.control_input = self.controller.run(self.state, self.Idx) # TODO : Replace this with a state estimate
                assert(len(self.control_input) == (self.num_RWs + self.num_MTBs))
                last_controller_update = sim_delta_time

            # Echo the Heartbeat once every 1000s
            if sim_delta_time - last_print_time >= 1000:
                print(f"Heartbeat: {sim_delta_time}")
                last_print_time = sim_delta_time

            # Step through the sim
            self.step(sim_delta_time, self.params.dt)

            sim_delta_time += self.params.dt
        
        # Report the sim speed-up
        elapsed_seconds_wall_clock = time() - WALL_START_TIME
        speed_up = self.params.MAX_TIME / elapsed_seconds_wall_clock
        print(
            f'Sim ran {speed_up:.4g}x faster than realtime. Took {elapsed_seconds_wall_clock:.1f} [s] "wall-clock" to simulate {self.params.MAX_TIME} [s]'
        )
        

# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes

if __name__ == "__main__":
    TRIAL_DIRECTORY = os.environ["TRIAL_DIRECTORY"]
    PARAMETER_FILEPATH = os.environ["PARAMETER_FILEPATH"]
    TRIAL_NUMBER = int(os.environ["TRIAL_NUMBER"])
    sim = Simulator(TRIAL_NUMBER, TRIAL_DIRECTORY, PARAMETER_FILEPATH)
    print("Initialized")
    sim.run()
