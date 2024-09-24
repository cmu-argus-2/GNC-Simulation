# built-in imports
import yaml
import numpy as np
import datetime
from scipy.spatial.transform import Rotation as R

# file imports
from world.physics.dynamics import Dynamics
from simulation_manager import logger
from time import time
import os
from sensors.imu import IMU, Gyro, Accel
from sensors.star_tracker import StarTracker

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

        # Intialize the dynamics class as the "world"
        self.world = Dynamics(self.config)

        # Initialize state and date
        self.world_state = None
        self.measurements = None
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

    def run(self, log_directory):
        l = logger.MultiFileLogger(log_directory)

        dt = 1 / self.update_rate

        gyro = Gyro(
            np.deg2rad([3, 7, 2]),  # [rad/s]
            np.deg2rad([0.2, 0.1, 0.4]),  # [(rad/s)/sqrt(Hz)]
            [1e-3, 1e-2, -1e-4],  # [-]
            dt,  # [s]
        )

        ST = StarTracker(0.1)

        i = 0
        while self.date - self.config["mission"]["start_date"] <= self.config[
            "mission"
        ]["duration"] / (24 * 60 * 60):
            self.world_state = self.world.state

            # self.measurement = self.sensors(self.date, self.world_state) # UNCOMMENT WHEN SENSOR MODELS ARE IMPLEMENTED
            # self.control_inputs = self.controller(self.date, self.measurement)
            # self.control_torques = self.actuators(self.date, self.measurement)

            self.world.update(input=np.zeros((9,)))

            self.date += dt / (24 * 60 * 60)  # 1 second into Julian date conversion
            if i % 1000 == 0:
                print(i, self.date, self.world_state)
            l.log_v(
                "state_true.bin",
                self.date,
                self.world_state[:-6],
                "date [days]",
                [
                    *["x [m]", "y [m]", "z [m]"],
                    *["x [m/s]", "y [m/s]", "z [m/s]"],
                    *["x", "y", "z", "w"],
                    *["x [rad/s]", "y [rad/s]", "z [rad/s]"],
                ],
            )

            true_omega = self.world_state[10:13]
            meas_omega = gyro.get_measurement(true_omega)
            l.log_v(
                "omega_meausured.bin",
                self.date,
                meas_omega,
                "date [days]",
                ["x [rad/s]", "y [rad/s]", "z [rad/s]"],
            )

            # TODO check quaternion order
            ECI_R_BODY = R.from_quat(self.world_state[6:10])

            J2000_R_ECI = R.identity()  # TODO compute this
            BODY_R_ST = R.identity()  # TODO read this in from config

            true_J2000_R_ST = J2000_R_ECI * ECI_R_BODY * BODY_R_ST
            measured_J2000_R_ST = ST.get_measurement(true_J2000_R_ST)
            l.log_v(
                "J2000_R_ST_true.bin",
                self.date,
                true_J2000_R_ST.as_quat(),
                "date [days]",
                ["x", "y", "z", "w"],
            )
            l.log_v(
                "J2000_R_ST_measured.bin",
                self.date,
                measured_J2000_R_ST.as_quat(),
                "date [days]",
                ["x", "y", "z", "w"],
            )

            i += 1


# ANSI escape sequences for colored terminal output  (from ChatGPT)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
RESET = "\033[0m"  # Resets all attributes

if __name__ == "__main__":
    REPO_NAME = "GNC-Simulation"

    # "_rel" postfix indicates a relative filepath
    repo_root_rel = "."  # path to "REPO_NAME/"

    # paths relative to repo_root_rel:
    montecarlo_rel = "montecarlo/"

    # "_abs" suffix indicates an absolute filepath
    repo_root_abs = os.path.realpath(repo_root_rel)
    results_directory_abs = os.path.join(repo_root_abs, montecarlo_rel, "results/")
    os.system(f"mkdir -p {results_directory_abs}")
    # ensure repo_root_abs actually points to the REPO_NAME
    assert os.path.basename(repo_root_abs) == REPO_NAME

    # ensure paths exist
    assert os.path.exists(repo_root_abs), f"Nonexistent: {repo_root_abs}"
    assert os.path.exists(
        results_directory_abs
    ), f"Nonexistent: {results_directory_abs}"

    job_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(GREEN + f'job_name:{RESET} "{job_name}"')
    job_directory_abs = os.path.join(results_directory_abs, job_name)
    os.system(f"mkdir -p {job_directory_abs}")

    START = time()
    simulator = Simulator()
    simulator.run(job_directory_abs)
    END = time()
    print(f'Took {(END-START):.2f} seconds to simulate job "{job_name}"')
