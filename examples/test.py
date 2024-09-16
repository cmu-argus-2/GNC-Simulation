#!/usr/bin/env python3

import numpy as np
from argus2sim.sim import CubeSatSim


MISSION_DURATION = 60 * 60 * 24

step_config = {
    "step_position_and_velocity": {
        "enabled": True,
        "timestep": 60.0,
    },
    "step_attitude_and_angular_velocity": {
        "enabled": True,
        "timestep": 0.05,
    },
}

sim = CubeSatSim(config=step_config)
mu = 3.9860044188 * 10**14
r = 6.9 * 10**6
sim.position = np.array([0., 0., r,])
sim.velocity = np.array([np.sqrt(mu/r), 0., 0.,])
sim.angular_velocity = np.array([0.2, 0., 0.,])

print(f"Initial position: {sim.position}")
print(f"Initial attitude: {sim.attitude}")

for i in range(int(MISSION_DURATION / sim.timestep)):
    sim.step()

print(f"Final position: {sim.position}")
print(f"Final attitude: {sim.attitude}")
print(f"Final attitude 2-norm: {np.linalg.norm(sim.attitude, ord=2)}")
print(f"Current r over original r: {1/r * np.linalg.norm(sim.position, ord=2)}")
