# Main entry point for each trial in a Python Job
import argusim

# Pybind Exposed Functions
from argusim.build.world.pyphysics import rk4
from argusim.build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from argusim.build.sensors.pysensors import readSensors

# Python Imports
from argusim.simulation_manager import MultiFileLogger
from argusim.FSW.controllers.controller import Controller
from argusim.FSW.estimators.AttitudeEKF import Attitude_EKF
from argusim.actuators import Magnetorquer
from argusim.actuators import ReactionWheel
from argusim.sensors.Sensor import SensorNoiseParams, TriAxisSensor
from argusim.sensors.SunSensor import SunSensor
from argusim.sensors.Bias import BiasParams

import os
import yaml
import numpy as np
from time import time
from scipy.spatial.transform import Rotation as R


attitude_estimate_error_labels = [f"{axis} [rad]" for axis in "xyz"]
gyro_bias_error_labels = [f"{axis} [rad/s]" for axis in "xyz"]
true_gyro_bias_labels = [f"{axis} [rad/s]" for axis in "xyz"]
EKF_sigma_labels = [f"attitude error {axis} [rad]" for axis in "xyz"] + [
    f"gyro bias error {axis} [rad/s]" for axis in "xyz"
]
EKF_state_labels = [f"q_{component}" for component in "wxyz"] + [f"{axis} [rad/s]" for axis in "xyz"]


class Simulator:
    def __init__(self, trial_number, log_directory, config_path) -> None:
        np.random.seed(TRIAL_NUMBER)  # determinism

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
            self.obsw_params = yaml.safe_load(f)
        self.define_controller()

        # Initialization
        self.true_state = np.array(self.params.initial_true_state)
        self.J2000_start_time = self.params.sim_start_time
        self.control_input = np.zeros((self.params.num_MTBs + self.params.num_RWs))

        # Gyro config
        initial_bias_range = np.array(self.obsw_params["gyroscope"]["initial_bias_range"])  # [rad/s]
        sigma_v_range = np.array(self.obsw_params["gyroscope"]["gyro_sigma_v_range"]) # [rad/sqrt(s)]
        sigma_w_range = np.array(self.obsw_params["gyroscope"]["gyro_sigma_w_range"]) # [(rad/s)/sqrt(s))]
        scale_factor_error_range = np.array(self.obsw_params["gyroscope"]["gyro_scale_factor_err_range"])  # [-]
        gyro_dt = self.params.gyro_dt

        gyro_params = []
        for i in range(3):
            biasParams = BiasParams.get_random_params(initial_bias_range, sigma_w_range)
            gyro_params.append(SensorNoiseParams.get_random_params(biasParams, sigma_v_range, scale_factor_error_range))
        self.gyro = TriAxisSensor(gyro_dt, gyro_params)

        # Sun Sensor config
        sigma_sunsensor = self.obsw_params["photodiodes"]["sigma_sunsensor"]
        photodiodes_dt  = self.params.photodiodes_dt
        self.sunSensor  = SunSensor(photodiodes_dt, sigma_sunsensor)

        # Magnetometer config
        mag_params = []
        for i in range(3):
            biasParams = BiasParams.get_random_params(initial_bias_range, sigma_w_range)
            mag_params.append(SensorNoiseParams.get_random_params(biasParams, 0.0*sigma_v_range, 0.0*scale_factor_error_range))
        magnetometer_dt   = self.params.magnetometer_dt
        self.magnetometer = TriAxisSensor(magnetometer_dt, mag_params)

        # Attitude Estimator Config
        self.attitude_ekf = Attitude_EKF(
            np.array(self.obsw_params["MEKF"]["sigma_initial_attitude"]),
            np.array(self.obsw_params["MEKF"]["sigma_gyro_white"]),
            np.array(self.obsw_params["MEKF"]["sigma_gyro_bias_deriv"]),
        )

        # Logging
        self.logr = MultiFileLogger(log_directory)
        self.state_labels = [
            "r_x ECI [m]",
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
            "zMag ECI [T]",
        ] + ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.num_RWs)]

        self.measurement_labels = [
            "gps_posx ECEF [m]",
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
            "mag_z_body [T]",
        ] + ["light_sensor_lux " + str(i) for i in range(self.num_photodiodes)]

        self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] + [
            "T_RW_" + str(i) + " [Nm]" for i in range(self.num_RWs)
        ]

    def define_estimator(self):
        """
        Defines the attitude estimator related parameters
        """
        self.Idx["NY"] = 12+self.num_photodiodes+self.num_RWs
        self.Idx["Y"] = dict()
        self.Idx["Y"]["GPS_POS"] = slice(0, 3)
        self.Idx["Y"]["GPS_VEL"] = slice(3, 6)
        self.Idx["Y"]["GYRO"] = slice(6, 9)
        self.Idx["Y"]["MAG"] = slice(9, 12)
        self.Idx["Y"]["SUN"] = slice(12, 12+self.num_photodiodes)
        self.Idx["Y"]["RW_OMEGA"] = slice(12+self.num_photodiodes, 12+self.num_photodiodes+self.num_RWs)

    def define_controller(self):
        """
        Defines the world for the controller objects
        """
        self.Idx = {}
        self.Idx["NX"] = 19
        self.Idx["X"] = dict()
        self.Idx["X"]["ECI_POS"] = slice(0, 3)
        self.Idx["X"]["ECI_VEL"] = slice(3, 6)
        self.Idx["X"]["TRANS"] = slice(0, 6)
        self.Idx["X"]["QUAT"] = slice(6, 10)
        self.Idx["X"]["ANG_VEL"] = slice(10, 13)
        self.Idx["X"]["ROT"] = slice(6, 13)
        self.Idx["X"]["SUN_POS"] = slice(13, 16)
        self.Idx["X"]["MAG_FIELD"] = slice(16, 19)

        # Actuator specific data
        # self.ReactionWheels = [ReactionWheel(self.config, IdRw) for IdRw in range(self.config["satellite"]["N_rw"])]

        # Actuator Indexing
        self.num_RWs = self.obsw_params["reaction_wheels"]["N_rw"]
        self.num_MTBs = self.obsw_params["magnetorquers"]["N_mtb"]
        self.Idx["NU"] = self.num_RWs + self.num_MTBs
        self.Idx["N_rw"] = self.num_RWs
        self.Idx["N_mtb"] = self.num_MTBs
        self.Idx["U"] = dict()
        self.Idx["U"]["MTB_TORQUE"] = slice(0, self.num_MTBs)
        self.Idx["U"]["RW_TORQUE"] = slice(self.num_MTBs, self.num_RWs + self.num_MTBs)
        # RW speed should be a state because it depends on the torque applied and needs to be propagated
        self.Idx["NX"] = self.Idx["NX"] + self.num_RWs
        self.Idx["X"]["RW_SPEED"] = slice(19, 19 + self.num_RWs)

        Magnetorquers = [Magnetorquer(self.obsw_params["magnetorquers"], IdMtb) for IdMtb in range(self.num_MTBs)]
        ReactionWheels = [
            ReactionWheel(self.obsw_params["reaction_wheels"], IdRw) for IdRw in range(self.num_RWs)
        ]
        self.controller = Controller(self.obsw_params, Magnetorquers, ReactionWheels, self.Idx)

        # Controller Frequency
        self.controller_dt = self.obsw_params["controller_dt"]

        # sensor sampling frequency
        self.last_sun_sensor_measurement_time = 0.0
        self.last_magnetometer_measurement_time = 0.0
        self.last_gyro_measurement_time = 0.0

    def set_control_input(self, u):
        """
        Sets the control input field of the class
        Exists for FSW to provide control inputs
        """
        if len(u) < len(self.control_input) - 1:
            raise Exception("Control Input not provided to all Magnetorquers")
        elif len(u) == len(self.control_input) - 1:
            self.control_input[0 : len(u)] = u  # Only magnetorquers
        else:
            self.control_input = u  # magnetorquers + RWs

    def sensors(self, current_time, state):
        """
        Implements partial observability using sensor models
        """
        return readSensors(state, current_time, self.params)

    def step(self, sim_time, dt):
        """
        Executes a single simulation step of a given step size
        This function is written separately to allow FSW to access simualtion stepping
        """
        # Time
        current_time = self.J2000_start_time + sim_time

        # Get control input
        control_input = self.control_input

        # Step through the simulation
        self.true_state = rk4(self.true_state, control_input, self.params, current_time, dt)

        # Mask state through sensors
        measurement = self.sensors(current_time, self.true_state)

        true_ECI_R_body = R.from_quat([*self.true_state[7:10], self.true_state[6]])

        # Sun Sensor update
        got_sun = False
        SUN_IN_VIEW = True  # TODO actually check if sun is in view
        if SUN_IN_VIEW and (current_time >= self.last_sun_sensor_measurement_time + self.sunSensor.dt):
            
            # TODO simulate RTC and use its drifting time
            true_sun_ray_ECI = self.true_state[self.Idx["X"]["SUN_POS"]]
            true_sun_ray_ECI /= np.linalg.norm(true_sun_ray_ECI)
            true_sun_ray_body = true_ECI_R_body.inv().as_matrix() @ true_sun_ray_ECI
            
            measured_sun_ray_in_body = self.sunSensor.get_measurement(true_sun_ray_body)
            self.attitude_ekf.sun_sensor_update(
                measured_sun_ray_in_body, true_sun_ray_ECI, current_time, self.sunSensor.sigma_angular_error
            )
            self.last_sun_sensor_measurement_time = current_time
            self.logr.log_v(
                "sun_sensor_measurement.bin",
                [current_time - self.J2000_start_time] + measured_sun_ray_in_body.tolist(),
                ["Time [s]"] + [f"{axis} [-]" for axis in "xyz"],
            )
            got_sun = True

        got_B = False
        if current_time >= self.last_magnetometer_measurement_time + self.magnetometer.dt:
            true_Bfield_ECI = self.true_state[self.Idx["X"]["MAG_FIELD"]]
            true_Bfield_ECI /= np.linalg.norm(true_Bfield_ECI)
            measured_Bfield_in_body = measurement[9:12]

            self.attitude_ekf.Bfield_update(
                measured_Bfield_in_body, true_Bfield_ECI, current_time, self.params.magnetometer_direction_noise_std
            )
            self.last_magnetometer_measurement_time = current_time
            self.logr.log_v(
                "magnetometer_measurement.bin",
                [current_time - self.J2000_start_time] + measured_Bfield_in_body.tolist(),
                ["Time [s]"] + [f"{axis} [T]" for axis in "xyz"],
            )
            got_B = True

        if got_B and got_sun:
            if not self.attitude_ekf.initialized:
                attitude_estimate = self.attitude_ekf.triad(
                    true_sun_ray_ECI, measured_sun_ray_in_body, true_Bfield_ECI, measured_Bfield_in_body
                )
                self.attitude_ekf.set_ECI_R_b(R.from_matrix(attitude_estimate))
                self.attitude_ekf.initialized = True
                self.attitude_ekf.P[0:3, 0:3] = np.eye(3) * np.deg2rad(10) ** 2
                self.attitude_ekf.P[3:6, 3:6] = np.eye(3) * np.deg2rad(5) ** 2

        # Propagate on Gyro
        if current_time >= self.last_gyro_measurement_time + self.gyro.dt:
            true_omega_body_wrt_ECI_in_body = self.true_state[10:13]
            gyro_measurement = self.gyro.update(true_omega_body_wrt_ECI_in_body)
            self.attitude_ekf.gyro_update(gyro_measurement, current_time)
            self.last_gyro_measurement_time = current_time

            self.logr.log_v(
                "gyro_measurement.bin",
                [current_time - self.J2000_start_time] + gyro_measurement.tolist(),
                ["Time [s]"] + [f"{axis} [rad/s]" for axis in "xyz"],
            )

        # Log pertinent Quantities

        true_gyro_bias = self.gyro.get_bias()

        if self.attitude_ekf.initialized:
            # get attitude estimate of the body wrt ECI
            estimated_ECI_R_body = self.attitude_ekf.get_ECI_R_b()
            attitude_estimate_error = (true_ECI_R_body * estimated_ECI_R_body.inv()).as_rotvec()

            estimated_gyro_bias = self.attitude_ekf.get_gyro_bias()
            gyro_bias_error = true_gyro_bias - estimated_gyro_bias

            self.logr.log_v(
                "EKF_state.bin",
                [current_time - self.J2000_start_time] + self.attitude_ekf.get_state().tolist(),
                ["Time [s]"] + EKF_state_labels,
            )
            self.logr.log_v(
                "EKF_error.bin",
                [current_time - self.J2000_start_time] + attitude_estimate_error.tolist() + gyro_bias_error.tolist(),
                ["Time [s]"] + attitude_estimate_error_labels + gyro_bias_error_labels,
            )

            EKF_sigmas = self.attitude_ekf.get_uncertainty_sigma()
            self.logr.log_v(
                "state_covariance.bin",
                [current_time - self.J2000_start_time] + EKF_sigmas.tolist(),
                ["Time [s]"] + EKF_sigma_labels,
            )

        self.logr.log_v(
            "gyro_bias_true.bin",
            [current_time - self.J2000_start_time] + true_gyro_bias.tolist(),
            ["Time [s]"] + true_gyro_bias_labels,
        )

        self.logr.log_v(
            "state_true.bin",
            [current_time - self.J2000_start_time]
            + self.true_state.tolist()
            + measurement.tolist()
            + control_input.tolist(),
            ["Time [s]"] + self.state_labels + self.measurement_labels + self.input_labels,
        )

        return measurement

    def run(self):
        """
        Runs the entire simulation
        Calls the 'step' function to run through the sim

        NOTE: This function operates on delta time i.e., seconds since mission start
              This is done to allow FSW to provide a delta time to step()
        """

        WALL_START_TIME = time()

        # Load Sim start times
        sim_delta_time = 0
        last_controller_update = 0
        last_print_time = 0

        # Run iterations
        while sim_delta_time <= self.params.MAX_TIME:

            # Update the controller
            if sim_delta_time >= last_controller_update + self.controller_dt:
                self.control_input = self.controller.run(
                    self.true_state, self.Idx
                )  # TODO : Replace this with a state estimate
                assert len(self.control_input) == (self.num_RWs + self.num_MTBs)
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
    TRIAL_NUMBER = int(os.environ["TRIAL_NUMBER"])
    TRIAL_DIRECTORY = os.environ["TRIAL_DIRECTORY"]
    PARAMETER_FILEPATH = os.environ["PARAMETER_FILEPATH"]
    sim = Simulator(TRIAL_NUMBER, TRIAL_DIRECTORY, PARAMETER_FILEPATH)
    print("Initialized")
    sim.run()
