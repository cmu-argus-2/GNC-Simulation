# Main entry point for each trial in a Python Job

# Pybind Exposed Functions
from argusim.build.world.pyphysics import rk4
from argusim.build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from argusim.build.sensors.pysensors import readSensors

# Python Imports
import os
from argusim.simulation_manager import SimLogger
from argusim.simulation_manager.indexing import IDX
import numpy as np
from argusim.world.LUT_generator import generate_lookup_tables
import yaml


class Simulator():
    def __init__(self, trial_number, log_directory, config_path, log=True) -> None:
        
        self.trial_number = trial_number

        # Datapaths
        self.config_path   = config_path
        self.log_directory = log_directory
        self.log = log

        # [TODO:] remove the next ~10 lines 
        with open(config_path, "r") as f:
            self.obsw_params = yaml.safe_load(f)

        # if data_path does not, 
        if self.obsw_params["useLUTs"]:
            data_path = os.path.realpath("./argusim/data/lookup_tables.yaml")
            if not os.path.exists(data_path):
                generate_lookup_tables(data_path)
        else:
            data_path = ""
        
        # Spacecraft Config
        self.params = SimParams(self.config_path, self.trial_number, self.log_directory, data_path)
        self.num_RWs = self.params.num_RWs
        self.num_MTBs = self.params.num_MTBs
        self.num_photodiodes = self.params.num_photodiodes
        self.num_panels = self.params.num_panels
        self.num_stk = self.params.num_stk

        # Initialization
        self.fsw_state = np.zeros((28,))
        self.state = np.array(self.params.initial_state)
        self.J2000_start_time = self.params.sim_start_time
        self.current_time = self.J2000_start_time
        self.control_input = np.zeros((self.params.num_MTBs + self.params.num_RWs + 1)) # MTBs + RW + Jetson ON?

        self.Idx = IDX(self.num_RWs, self.num_MTBs, self.num_stk, self.num_photodiodes, self.num_panels)

        percent_to_log = self.obsw_params["PlotFlags"]["percent_to_log"]
        self.log_counter = 0
        self.log_interval = int(1 / percent_to_log) if percent_to_log > 0 else 1

        # Logging
        if self.log:
            self.logr = SimLogger(log_directory, self.num_RWs, self.num_stk, self.num_photodiodes, self.num_MTBs, self.num_panels, self.J2000_start_time)

    
    def set_control_input(self, u):
        '''
            Sets the control input field of the class
            Exists for FSW to provide control inputs
        '''
        if len(u) < self.num_MTBs:
            raise Exception("Control Input not provided to all Magnetorquers")
        elif len(u) == self.num_MTBs:
            self.control_input[0:len(u)] = u # Only magnetorquers
        elif len(u) == self.num_MTBs + self.num_RWs:
            self.control_input[0:len(u)] = u # Only magnetorquers + RW
        else:
            self.control_input = u # magnetorquers + RWs + Jetson
    
    def sensors(self, current_time, state, control_input):
        '''
            Implements partial observability using sensor models
        '''
        measurement = readSensors(state, control_input, current_time, self.params)
        measurement = np.array(measurement)
        return measurement
    
    def get_time(self):
        '''
            Get current simulation time
        '''
        return self.current_time
    
    def get_indexes(self):
        """
        Returns the index dictionary for state, control input, and measurement vectors.
        """
        return self.Idx

    def step(self, sim_time, dt):
        """
        Executes a single simulation step of a given step size
        This function is written separately to allow FSW to access simualtion stepping
        """
        ## Real World
        # Time
        self.current_time = self.J2000_start_time + sim_time
        
        # Get control input
        control_input = self.control_input

        # Step through the simulation
        self.state = rk4(self.state, control_input, self.params, self.current_time, dt)
        
        # Mask state through sensors
        measurement = self.sensors(self.current_time + dt, self.state, control_input)
        
        # Log pertinent Quantities
        if self.log and self.log_counter % self.log_interval == 0:
            # Log true state
            self.logr.log_true_state(self.current_time + dt, self.state, control_input)
            # measurement data logging
            self.logr.log_measurements(self.current_time + dt, measurement)
            # Log fsw state
            self.logr.log_fsw_state(self.current_time + dt, self.fsw_state, control_input)

            self.log_counter = 1
        else:
            self.log_counter += 1

        return measurement
