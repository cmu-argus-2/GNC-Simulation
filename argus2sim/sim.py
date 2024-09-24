import math
import numpy as np
from argus2sim.bin.dynamics import step_position_and_velocity,\
                                   step_attitude_and_angular_velocity


class CubeSatSim():
    def __init__(self, config: dict):
        self.timestep = self.get_gcd_timestep(config)
        self.iter_tracker = self.init_iter_tracker(config, self.timestep)
        self.config = config
        self.position = np.zeros(3)
        self.attitude = np.array([1., 0., 0., 0.])
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.body_frame_torque = np.zeros(3)

    def float_gcd(self, a: float, b: float, tol=10**-6) -> float:
        if a < b:
            return self.float_gcd(b, a)
        if abs(b) < tol:
            return a
        else :
            return self.float_gcd(b, a - math.floor(a / b) * b)

    def get_gcd_timestep(self, config: dict) -> float:
        gcd_timestep = list(config.values())[0]["timestep"]
        for substep_id, substep_config in config.items():
            if not substep_config["enabled"]:
                continue
            else:
                gcd_timestep = self.float_gcd(gcd_timestep, substep_config["timestep"])
        return gcd_timestep

    def init_iter_tracker(self, config: dict, gcd_timestep: float) -> dict:
        iter_tracker = {}
        for substep_id, substep_config in config.items():
            if not substep_config["enabled"]:
                continue
            else:
                iter_tracker[substep_id] = {
                    "max_iter": int(round(substep_config["timestep"] / gcd_timestep)),
                    "curr_iter": 0,
                }
        return iter_tracker

    def step(self) -> None:
        for substep_id, substep_config in self.config.items():
            if not substep_config["enabled"]:
                continue
            else:
                self.iter_tracker[substep_id]["curr_iter"] += 1
                if self.iter_tracker[substep_id]["curr_iter"] == self.iter_tracker[substep_id]["max_iter"]:
                    self.iter_tracker[substep_id]["curr_iter"] = 0
                    substep = getattr(self, substep_id)
                    substep(substep_config["timestep"])

    def step_position_and_velocity(self, dt: float):
        pos_and_vel = step_position_and_velocity(
            np.hstack((self.position, self.velocity)),
            dt,
        )
        self.position = pos_and_vel[:self.position.shape[0]]
        self.velocity = pos_and_vel[-self.velocity.shape[0]:]

    def step_attitude_and_angular_velocity(self, dt: float):
        att_and_ang_vel = step_attitude_and_angular_velocity(
            np.hstack((self.attitude, self.angular_velocity)),
            self.body_frame_torque,
            dt
        )
        self.attitude = att_and_ang_vel[:self.attitude.shape[0]]
        self.angular_velocity = att_and_ang_vel[-self.angular_velocity.shape[0]:]
