import numpy as np
from Sensor import TriAxisSensor


class IMUNoiseParams:
    def __init__(self, gyro_params, accel_params):
        """Gyroscope and Accelerometer parameters

        Args:
            gyro_params ([SensorNoiseParams x 3]): list of SensorNoiseParams, one per x, y, z axes
            accel_params ([SensorNoiseParams x 3]): list of SensorNoiseParams, one per x, y, z axes
        """
        self.gyro_params = gyro_params
        self.accel_params = accel_params


class IMU:
    def __init__(self, dt, IMU_noise_params):
        self.gyro = TriAxisSensor(dt, IMU_noise_params.gyro)
        self.accel = TriAxisSensor(dt, IMU_noise_params.accel)

    def get_bias(self):
        gyro_bias = self.gyro.get_bias()
        accel_bias = self.accel.get_bias()
        return gyro_bias, accel_bias

    def update(self, clean_gyro_signal, clean_accel_signal):
        gyro_measurement = self.gyro.update(clean_gyro_signal)
        accel_measurement = self.accel.update(clean_accel_signal)
        return gyro_measurement, accel_measurement



class IMU:
    def __init__(self, dt, IMU_noise_params):
        self.gyro = TriAxisSensor(dt, IMU_noise_params.gyro)
        self.accel = TriAxisSensor(dt, IMU_noise_params.accel)

    def get_bias(self):
        gyro_bias = self.gyro.get_bias()
        accel_bias = self.accel.get_bias()
        return gyro_bias, accel_bias

    def update(self, clean_gyro_signal, clean_accel_signal):
        gyro_measurement = self.gyro.update(clean_gyro_signal)
        accel_measurement = self.accel.update(clean_accel_signal)
        return gyro_measurement, accel_measurement
