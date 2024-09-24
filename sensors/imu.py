import numpy as np


class InertialSensor:
    def __init__(self, initial_bias, sigma, scale_factor_error, dt):
        """_summary_

        Args:
            initial_bias (double):                [units]
            sigma (double): noise                 [units/sqrt(Hz)]
            scale_factor_error (double):          [-]
            dt (double):                          [s]
        """
        self.bias = initial_bias
        self.sigma_discrete = sigma / np.sqrt(dt)
        self.scale_factor_error = scale_factor_error
        self.dt = dt

    def get_measurement(self, true_value):
        noise = np.random.normal(0, self.sigma_discrete)
        return (1 + self.scale_factor_error) * true_value + self.bias + noise


class TriAxialInertialSensor:
    def __init__(self, initial_bias, sigma, scale_factor_error, dt):
        """_summary_

        Args:
            initial_bias (3x1):        [units]
            sigma (3x1): noise         [units/sqrt(Hz)]
            scale_factor_error (3x1):  [-]
            dt (double):               [s]
        """
        self.x = InertialSensor(initial_bias[0], sigma[0], scale_factor_error[0], dt)
        self.y = InertialSensor(initial_bias[1], sigma[1], scale_factor_error[1], dt)
        self.z = InertialSensor(initial_bias[2], sigma[2], scale_factor_error[2], dt)

    def get_measurement(self, true_value):
        # true_value (3x1): [units]
        return np.array(
            [
                self.x.get_measurement(true_value[0]),
                self.y.get_measurement(true_value[1]),
                self.z.get_measurement(true_value[2]),
            ]
        )


class Gyro(TriAxialInertialSensor):
    def __init__(self, initial_bias, sigma_ARW, scale_factor_error, dt):
        """_summary_

        Args:
            initial_bias (3x1):                [rad/s]
            sigma_ARW (3x1): angle random walk [(rad/s)/sqrt(Hz)]
            scale_factor_error (3x1):          [-]
            dt (double):                       [s]
        """
        super().__init__(initial_bias, sigma_ARW, scale_factor_error, dt)

    def get_measurement(self, true_omega):
        return super().get_measurement(true_omega)


class Accel(TriAxialInertialSensor):
    def __init__(self, initial_bias, sigma_VRW, scale_factor_error, dt):
        """_summary_

        Args:
            initial_bias (3x1):                   [m/s^2]
            sigma_VRW (3x1): velocity random walk [(m/s^2)/sqrt(Hz)]
            scale_factor_error (3x1):             [-]
            dt (double):                          [s]
        """
        super.__init__(initial_bias, sigma_VRW, scale_factor_error, dt)

    def get_measurement(self, true_accel):
        return super.get_measurement(true_accel)


class IMU:
    def __init__(self, gyro, accel):
        self.gyro = gyro
        self.accel = accel

    def get_measurement(self, true_omega, true_accel):
        return (
            self.gyro.get_measurement(true_omega),
            self.accel.get_measurement(true_accel),
        )
