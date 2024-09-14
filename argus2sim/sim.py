import json

import casadi as cs
import numpy as np

from argus2sim.config import sim_config


class CubeSatSim():
    def __init__(self):
        self.init_members()

    def step(self) -> None:
        for method_name in sim_config["method_order"]:
            if not sim_config["methods"][method_name]["enabled"]:
                continue
            else:
                substep = getattr(self, method_name)
                substep()

    def init_members(self) -> None:
        member_config = sim_config["members"]
        for member_name, member_config in sim_config["members"].items():
            if not member_config["enabled"]:
                continue
            else:
                if member_config["len"] == 1:
                    setattr(self, name, 0.0)
                elif member_config["len"] > 1:
                    setattr(self, member_name, np.zeros(member_config["len"]))

    def step_position_and_velocity(self):
        print(self.position_and_velocity)

    def step_attitude_and_angular_velocity(self):
        print(self.attitude_and_angular_velocity)
