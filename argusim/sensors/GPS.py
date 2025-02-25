import numpy as np

class GPS():
    def __init__(self, dt):
        self.dt   = dt
        self.last_meas_time = -np.inf
        self.last_measurement = np.zeros(6)

    