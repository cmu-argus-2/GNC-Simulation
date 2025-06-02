# Pybind Exposed Functions
from argusim.build.world.pyphysics import rk4
from argusim.build.simulation_utils.pysim_utils import Simulation_Parameters as SimParams
from argusim.build.sensors.pysensors import readSensors

# Python Imports
import os
from argusim.simulation_manager import MultiFileLogger
import numpy as np
from argusim.world.LUT_generator import generate_lookup_tables
from time import time
from argusim.simulation_manager.sim import Simulator

def run(sim):
    """
    Runs the entire simulation
    Calls the 'step' function to run through the sim

    NOTE: This function operates on delta time i.e., seconds since mission start
            This is done to allow FSW to provide a delta time to step()
    """

    WALL_START_TIME = time()

    # Load Sim start times
    sim_delta_time = 0
    last_print_time = 0

    # Run iterations
    while sim_delta_time <= sim.params.MAX_TIME:
        sim.set_control_input(np.zeros(sim.Idx["NU"]))

        # Echo the Heartbeat once every 1000s
        if sim_delta_time - last_print_time >= 1000:
            print(f"Heartbeat: {sim_delta_time}")
            last_print_time = sim_delta_time

        # Step through the sim
        sim.step(sim_delta_time, sim.params.dt)

        sim_delta_time += sim.params.dt
    # Report the sim speed-up
    elapsed_seconds_wall_clock = time() - WALL_START_TIME
    speed_up = sim.params.MAX_TIME / elapsed_seconds_wall_clock
    print(
        f'Sim ran {speed_up:.4g}x faster than realtime. Took {elapsed_seconds_wall_clock:.1f} [s] "wall-clock" to simulate {sim.params.MAX_TIME} [s]'
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
    run(sim)