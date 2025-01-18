from math import sqrt
import numpy as np
from .Bias import Bias


class SensorNoiseParams:
    def __init__(self, biasParams, sigma_v, scale_factor_error):
        """Parameters for a time-varying bias modeled as a random walk

        Args:
            biasParams (BiasParams): bias parameters
            sigma_w (float): continuous-time power spectral density of additive white noise to sensor output [units/sqrt(Hz)]
            scale_factor_error (float): multiplier [-]
        """
        self.bias_params = biasParams
        self.sigma_v = sigma_v
        self.scale_factor_error = scale_factor_error

    def get_random_params(biasParams, sigma_v_range, scale_factor_error_range):
        return SensorNoiseParams(
            biasParams, np.random.uniform(*sigma_v_range), np.random.uniform(*scale_factor_error_range)
        )


class Sensor:
    def __init__(self, dt, sensor_noise_params):
        self.dt   = dt
        self.bias = Bias(dt, sensor_noise_params.bias_params)

        # discrete version of sensor_noise_params.sigma_v causing the bias to random walk when integrated
        self.white_noise = sensor_noise_params.sigma_v / sqrt(dt)

        self.scale_factor_error = sensor_noise_params.scale_factor_error

    def update(self, clean_signal):
        self.bias.update()
        noise = self.white_noise * np.random.standard_normal()
        return (1 + self.scale_factor_error) * clean_signal + self.bias.get_bias() + noise

    def get_bias(self):
        return self.bias.get_bias()


class TriAxisSensor:
    def __init__(self, dt, axes_params):
        self.dt = dt
        self.x  = Sensor(dt, axes_params[0])
        self.y  = Sensor(dt, axes_params[1])
        self.z  = Sensor(dt, axes_params[2])

    def get_bias(self):
        return np.array([self.x.get_bias(), self.y.get_bias(), self.z.get_bias()])

    def update(self, clean_signal):
        return np.array(
            [self.x.update(clean_signal[0]), self.y.update(clean_signal[1]), self.z.update(clean_signal[2])]
        )
