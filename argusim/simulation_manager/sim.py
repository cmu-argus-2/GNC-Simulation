# Main entry point for each trial in a Python Job

# Pybind Exposed Functions
from argusim.build.world.pyphysics import rk4
from argusim.build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from argusim.build.sensors.pysensors import readSensors

# Python Imports
import os
from argusim.simulation_manager import MultiFileLogger
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

        # Initialization
        self.state = np.array(self.params.initial_state)
        self.fsw_state = np.zeros((28,))
        self.J2000_start_time = self.params.sim_start_time
        self.current_time = self.J2000_start_time
        self.control_input = np.zeros((self.params.num_MTBs + self.params.num_RWs + 1)) # MTBs + RW + Jetson ON?

        self.define_indexes()

        # Logging
        if self.log:
            self.logr = MultiFileLogger(log_directory)
            # [TODO:] remove redundant definition, already in the logger class
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
                                ["bias_x [rad/s]",
                                "bias_y [rad/s]",
                                "bias_z [rad/s]"]    + \
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
                                       ['mtb_power ' + str(i) for i in range(self.num_MTBs)] +\
                                       ['solar_power' + str(i) for i in range(self.num_panels)] +\
                                       ["Battery SoC [%]", "Battery Capacity [J]", "Battery Current [A]",
                                        "Battery Voltage [V]", "Battery Mid Voltage [V]", "Battery TTE [s]",
                                        "Battery TTF [s]", "Battery Temperature [K]"] + ["Jetson Power [W]"]
            
            self.fsw_labels = ["fsw_gps_posx ECI [m]", 
                                "fsw_gps_posy ECI [m]", 
                                "fsw_gps_posz ECI [m]", 
                                "fsw_gps_velx ECI [m/s]", 
                                "fsw_gps_vely ECI [m/s]", 
                                "fsw_gps_velz ECI [m/s]",
                                "fsw_qw",
                                "fsw_qx",
                                "fsw_qy",
                                "fsw_qz",
                                "fsw_gyro_x [rad/s]", 
                                "fsw_gyro_y [rad/s]", 
                                "fsw_gyro_z [rad/s]",
                                "fsw_bias_x [rad/s]",
                                "fsw_bias_y [rad/s]",
                                "fsw_bias_z [rad/s]", 
                                "fsw_mag_x_body [T]", 
                                "fsw_mag_y_body [T]", 
                                "fsw_mag_z_body [T]",
                                "fsw_sun_x",
                                "fsw_sun_y",
                                "fsw_sun_z",
                                "fsw_sun_eci_x",
                                "fsw_sun_eci_y",
                                "fsw_sun_eci_z",
                                "fsw_mag_eci_x",
                                "fsw_mag_eci_y",
                                "fsw_mag_eci_z"]
            
            self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] + \
                                ["T_RW_" + str(i) + " [Nm]" for i in range(self.num_RWs)] + ["Jetson ON"]

    def define_indexes(self):
        # # Indexing
        # State
        self.Idx = {}
        self.Idx["NX"] = 22 + self.num_RWs
        self.Idx["X"] = dict()
        self.Idx["X"]["ECI_POS"] = slice(0, 3)
        self.Idx["X"]["ECI_VEL"] = slice(3, 6)
        self.Idx["X"]["TRANS"] = slice(0, 6)
        self.Idx["X"]["QUAT"] = slice(6, 10)
        self.Idx["X"]["ANG_VEL"] = slice(10, 13)
        self.Idx["X"]["ROT"] = slice(6, 13)
        self.Idx["X"]["SUN_POS"] = slice(13, 16)
        self.Idx["X"]["MAG_FIELD"] = slice(16, 19)
        self.Idx["X"]["GYRO_BIAS"] = slice(19,22)
        self.Idx["X"]["RW_SPEED"] = slice(22, 22 + self.num_RWs)
        self.Idx["X"]["BAT"] = slice(22 + self.num_RWs, 26 + self.num_RWs)
        self.Idx["X"]["BAT_SOC"] = slice(22 + self.num_RWs, 23 + self.num_RWs)
        self.Idx["X"]["BAT_TEMP"] = slice(23 + self.num_RWs, 24 + self.num_RWs)
        self.Idx["X"]["BAT_VOLT"] = slice(24 + self.num_RWs, 25 + self.num_RWs)
        self.Idx["X"]["BAT_CUR"] = slice(25 + self.num_RWs, 26 + self.num_RWs)

        # Control Input
        self.num_RWs = self.obsw_params["reaction_wheels"]["N_rw"]
        self.num_MTBs = self.obsw_params["magnetorquers"]["N_mtb"]
        self.Idx["NU"] = self.num_RWs + self.num_MTBs
        self.Idx["N_rw"] = self.num_RWs
        self.Idx["N_mtb"] = self.num_MTBs
        self.Idx["U"] = dict()
        self.Idx["U"]["MTB_TORQUE"] = slice(0, self.num_MTBs)
        self.Idx["U"]["RW_TORQUE"] = slice(self.num_MTBs, self.num_RWs + self.num_MTBs)        

        # Measurements
        self.Idx["Y"] = dict()
        self.Idx["Y"]["GPS"] = slice(0, 6)
        self.Idx["Y"]["GPS_POS"] = slice(0, 3)
        self.Idx["Y"]["GPS_VEL"] = slice(3, 6)
        self.Idx["Y"]["GYRO"] = slice(6, 9)
        self.Idx["Y"]["MAG"] = slice(9, 12)
        ny = 12
        self.Idx["Y"]["SUN"] = slice(ny, ny+self.num_photodiodes)
        ny = ny+self.num_photodiodes
        self.Idx["Y"]["RW_OMEGA"] = slice(ny, ny+self.num_RWs)
        ny = ny+self.num_RWs
        self.Idx["Y"]["MTB_POW"] = slice(ny, ny+self.num_MTBs)
        ny = ny+self.num_MTBs
        self.Idx["Y"]["SOL_POW"] = slice(ny, ny+self.num_panels)
        ny = ny+self.num_panels
        self.Idx["Y"]["BATTERY"] = slice(ny, ny + 8)
        self.Idx["Y"]["BAT_SOC"] = slice(ny, ny + 1)
        self.Idx["Y"]["BAT_CAP"] = slice(ny + 1, ny + 2)
        self.Idx["Y"]["BAT_CUR"] = slice(ny + 2, ny + 3)
        self.Idx["Y"]["BAT_VOL"] = slice(ny + 3, ny +4)
        self.Idx["Y"]["BAT_MIDVOL"] = slice(ny + 4, ny + 5)
        self.Idx["Y"]["BAT_TTE"] = slice(ny + 5, ny + 6)
        self.Idx["Y"]["BAT_TTF"] = slice(ny + 6, ny + 7)
        self.Idx["Y"]["BAT_TEMP"] = slice(ny + 7, ny + 8)
        self.Idx["Y"]["JET_POW"] = slice(ny + 8, ny + 9)
        self.Idx["NY"] = ny + 9

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
        result = readSensors(state, control_input, current_time, self.params)
        measurement = np.array(result.measurement)
        new_state = np.array(result.state)
        return measurement, new_state
    
    def get_time(self):
        '''
            Get current simulation time
        '''
        return self.current_time

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
        measurement, self.state = self.sensors(self.current_time, self.state, control_input)
        
        # Log pertinent Quantities
        # [TODO:] use logging class functions
        if self.log:
            self.logr.log_v(
                "state_true.bin",
                [self.current_time] + self.state.tolist() + measurement.tolist() + control_input.tolist() + self.fsw_state.tolist(),
                ["Time [s]"] + self.state_labels + self.measurement_labels + self.input_labels + self.fsw_labels
            )

        return measurement
