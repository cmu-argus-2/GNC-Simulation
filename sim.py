# Main entry point for each trial in a Python Job

from build.world.pyphysics import rk4
from build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from build.sensors.pysensors import gps, sunSensor, magnetometer, gyroscope

import numpy as np
from simulation_manager import logger
import os

class Simulator():
    def __init__(self, log_directory, config_path) -> None:
        # Datapaths
        self.config_path = config_path
        self.log_directory = log_directory
        
        # Spacecraft Config
        self.params = SimParams(self.config_path)
        self.num_RWs = self.params.num_RWs
        self.num_MTBs = self.params.num_MTBs
        self.num_light_diodes = self.params.num_light_diodes

        # Initialization
        self.state = np.array(self.params.initial_state)
        self.J2000_start_time = self.params.earliest_sim_start_unix
        self.control_input = np.zeros((self.params.num_MTBs + self.params.num_RWs))

        # Logging
        self.logr = logger.MultiFileLogger(log_directory)
        self.state_labels = ["r_x ECI [m]", "r_y ECI [m]", "r_z ECI [m]", "v_x ECI [m/s]", "v_y ECI [m/s]", "v_z ECI [m/s]",
                    "q_w", "q_x", "q_y", "q_z", "omega_x [rad/s]", "omega_y [rad/s]", "omega_z [rad/s]"] + ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.num_RWs)]
        
        self.measurement_labels = ["gps_posx ECEF [m]", "gps_posy ECEF [m]", "gps_posz ECEF [m]", "gps_velx ECEF [m/s]", "gps_vely ECEF [m/s]", "gps_velz ECEF [m/s]",
                                   "gyro_x [rad/s]", "gyro_y [rad/s]", "gyro_z [rad/s]", "mag_x_body [T]", "mag_y_body [T]", "mag_z_body [T]"] + ['light_sensor_lux ' + str(i) for i in range(self.num_light_diodes)]
        
        self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] + ["V_RW_" + str(i) + " [V]" for i in range(self.num_RWs)]

    def set_control_input(self, u):
        '''
            Sets the 'private' control input field of the class
        '''
        if len(u) < len(self.control_input) - 1:
            raise Exception("Control Input not provided to all Magnetorquers")
        elif len(u) == len(self.control_input)-1:
            self.control_input[0:len(u)] = u # Only magnetorquers
        else:
            self.control_input = u # magnetorquers + RWs
    
    def sensors(self, current_time, state):
        '''
            Implements partial observability using sesnor models
        '''
        gps_measurements = gps(state[0:3], state[3:6], current_time, self.params)
        sun_sensor_measurements = sunSensor(state[6:10], current_time, self.params)
        magnetic_field_measurements = magnetometer(state[0:3], state[6:10], current_time, self.params)
        gyroscope_readings = gyroscope(state[10:13])

        return np.concatenate((gps_measurements, gyroscope_readings, magnetic_field_measurements, sun_sensor_measurements))

    def step(self, sim_time, dt):
        '''
            Executes a single simulation step of a given step size
        '''
        # Time
        current_time = self.J2000_start_time + sim_time
        
        # Get control input
        control_input = self.control_input

        # Step the simulation
        self.state = rk4(self.state, control_input, self.params, current_time, dt)

        # Mask state through sensors
        measurement = self.sensors(current_time, self.state)

        print(len(self.state), len(measurement), len(control_input))
        # Log pertinent Quantities
        self.logr.log_v( # TODO : add state estimate and measurement labels
            "state_true.bin",
            [current_time] + self.state.tolist() + measurement.tolist() + control_input.tolist(),
            ["Time [s]"] + self.state_labels + self.measurement_labels + self.input_labels
        )

        return measurement



# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes

if __name__ == "__main__":
    TRIAL_DIRECTORY = os.environ["TRIAL_DIRECTORY"]
    PARAMETER_FILEPATH = os.environ["PARAMETER_FILEPATH"]
    sim = Simulator(TRIAL_DIRECTORY, PARAMETER_FILEPATH)
    dt = 0.01
    for i in range(10000000):
        sim.step(i*dt, dt)
