from actuators.magnetorquer import Magnetorquers
import numpy as np

class Actuation:
    def __init__(self, config):
        # should be a list of elements of the class magnetorquer, instead of a class magnetorquers. 
        # We may want to consider 2 vs 3 magnetorquers and different orientations depending on the volume available on the satellite
        self.magnetorquers = Magnetorquers()
        self.reaction_wheels = []

    # def add_magnetorquer(self, magnetorquer):
    #     self.magnetorquers.append(magnetorquer)

    # def add_reaction_wheel(self, reaction_wheel):
    #     self.reaction_wheels.append(reaction_wheel)

    def run(self, actuator_cmd, world_state, Idx):
        # Placeholder function to apply actuation
        # This should be replaced with actual implementation
        act_output = np.zeros(Idx["NU"])
        induced_dipole_moment = 1 
        act_output["U"]["MTB_TORQUE"] = self.magnetorquers.get_applied_torque(induced_dipole_moment, world_state[Idx["X"]["MAG_FIELD"]])

        act_output["U"]["RW_TORQUE"] = self.reaction_wheels.get_applied_torque(actuator_cmd)
        
        return actuator_cmd