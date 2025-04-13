# Main entry point for each trial in a Python Job

# Pybind Exposed Functions
from argusim.build.world.pyphysics import rk4
from argusim.build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from argusim.build.sensors.pysensors import readSensors

# Python Imports
from argusim.simulation_manager import MultiFileLogger
import numpy as np

class Simulator():
    def __init__(self, trial_number, log_directory, config_path, log=True) -> None:
        
        self.trial_number = trial_number

        # Datapaths
        self.config_path = config_path
        self.log_directory = log_directory
        self.log = log
        
        # Spacecraft Config
        self.params = SimParams(self.config_path, self.trial_number, self.log_directory)
        self.num_RWs = self.params.num_RWs
        self.num_MTBs = self.params.num_MTBs
        self.num_photodiodes = self.params.num_photodiodes

        # Initialization
        self.state = np.array(self.params.initial_state)
        self.J2000_start_time = self.params.sim_start_time
        self.control_input = np.zeros((self.params.num_MTBs + self.params.num_RWs + 1)) # MTBs + RW + Jetson ON?

        # Logging
        if self.log:
            self.logr = MultiFileLogger(log_directory)
            self.state_labels = ["r_x ECI [m]", 
                                 "r_y ECI [m]", 
                                 "r_z ECI [m]", 
                                 "v_x ECI [m/s]", 
                                 "v_y ECI [m/s]", 
                                 "v_z ECI [m/s]",
                                 "q_w", 
                                 "q_x", 
                                 "q_y", 
                                 "q_z", 
                                 "omega_x [rad/s]", 
                                 "omega_y [rad/s]", 
                                 "omega_z [rad/s]", 
                                 "rSun_x ECI [m]",
                                 "rSun_y ECI [m]",
                                 "rSun_z ECI [m]",
                                 "xMag ECI [T]",
                                 "yMag ECI [T]",
                                 "zMag ECI [T]"] + \
                                ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.num_RWs)] + \
                                ["Battery SoC", "Battery temperature [K]", "Pack Voltage [V]", "Pack Current [A]"]
            
            self.measurement_labels = ["gps_posx ECEF [m]", 
                                       "gps_posy ECEF [m]", 
                                       "gps_posz ECEF [m]", 
                                       "gps_velx ECEF [m/s]", 
                                       "gps_vely ECEF [m/s]", 
                                       "gps_velz ECEF [m/s]",
                                       "gyro_x [rad/s]", 
                                       "gyro_y [rad/s]", 
                                       "gyro_z [rad/s]", 
                                       "mag_x_body [T]", 
                                       "mag_y_body [T]", 
                                       "mag_z_body [T]"] + \
                                       ['light_sensor_lux ' + str(i) for i in range(self.num_photodiodes)] + \
                                       ['mtb_power ' + str(i) for i in range(self.num_MTBs)]
            
            self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] + \
                                ["T_RW_" + str(i) + " [Nm]" for i in range(self.num_RWs)]

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
    
    def sensors(self, current_time, state, control_input):
        '''
            Implements partial observability using sensor models
        '''
        return readSensors(state, control_input, current_time, self.params)

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
        measurement = self.sensors(current_time, self.state, control_input)
        
        # Log pertinent Quantities
        if self.log:
            self.logr.log_v(
                "state_true.bin",
                [current_time] + self.state.tolist() + measurement.tolist() + control_input.tolist(),
                ["Time [s]"] + self.state_labels + self.measurement_labels + self.input_labels
            )

        return measurement
