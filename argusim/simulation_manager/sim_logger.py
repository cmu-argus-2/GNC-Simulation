from argusim.simulation_manager import MultiFileLogger

class SimLogger(MultiFileLogger):
    def __init__(self, log_directory, num_RWs, num_photodiodes, num_MTBs, J2000_start_time):
        super().__init__(log_directory)
        self.num_RWs = num_RWs
        self.num_photodiodes = num_photodiodes
        self.num_MTBs = num_MTBs
        self.J2000_start_time = J2000_start_time

        self.state_labels = [
            "r_x ECI [m]",
            "r_y ECI [m]",
            "r_z ECI [m]",
            "v_x ECI [m/s]",
            "v_y ECI [m/s]",
            "v_z ECI [m/s]",
            "q_w",
            "q_x",
            "q_y",
            "q_z",
            "omega_x [rad/s]",
            "omega_y [rad/s]",
            "omega_z [rad/s]",
            "rSun_x ECI [m]",
            "rSun_y ECI [m]",
            "rSun_z ECI [m]",
            "xMag ECI [T]",
            "yMag ECI [T]",
            "zMag ECI [T]",
        ] + ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.num_RWs)]

        self.measurement_labels = [
            "gps_posx ECEF [m]",
            "gps_posy ECEF [m]",
            "gps_posz ECEF [m]",
            "gps_velx ECEF [m/s]",
            "gps_vely ECEF [m/s]",
            "gps_velz ECEF [m/s]",
            "gyro_x [rad/s]",
            "gyro_y [rad/s]",
            "gyro_z [rad/s]",
            "mag_x_body [T]",
            "mag_y_body [T]",
            "mag_z_body [T]",
        ] + ["light_sensor_lux " + str(i) for i in range(self.num_photodiodes)] + [
            "rw_encoder_" + str(i) + " [rad/s]" for i in range(self.num_RWs)
        ]

        self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] + [
            "T_RW_" + str(i) + " [Nm]" for i in range(self.num_RWs)
        ]

        self.attitude_estimate_error_labels = [f"{axis} [rad]" for axis in "xyz"]
        self.gyro_bias_error_labels = [f"{axis} [rad/s]" for axis in "xyz"]
        self.true_gyro_bias_labels = [f"{axis} [rad/s]" for axis in "xyz"]
        self.EKF_sigma_labels = [f"attitude error {axis} [rad]" for axis in "xyz"] + [
            f"gyro bias error {axis} [rad/s]" for axis in "xyz"
        ]
        self.EKF_state_labels = [f"q_{component}" for component in "wxyz"] + [f"{axis} [rad/s]" for axis in "xyz"]


    def log_measurements(self, current_time, measurements, Idx, gotSensor):
        if gotSensor["GotSun"]:
            self.log_v(
                "sun_sensor_measurement.bin",
                [current_time - self.J2000_start_time] + measurements[Idx["Y"]["SUN"]].tolist(),
                ["Time [s]"] + [f"{axis} [-]" for axis in "xyz"],
            )

        if gotSensor["GotMag"]:
            self.log_v(
                "magnetometer_measurement.bin",
                [current_time - self.J2000_start_time] + measurements[Idx["Y"]["MAG"]].tolist(),
                ["Time [s]"] + [f"{axis} [T]" for axis in "xyz"],
            )

        if gotSensor["GotGyro"]:
            self.log_v(
                "gyro_measurement.bin",
                [current_time - self.J2000_start_time] + measurements[Idx["Y"]["GYRO"]].tolist(),
                ["Time [s]"] + [f"{axis} [rad/s]" for axis in "xyz"],
            )

    def log_true_state(self, current_time, true_state, control_input, sensor_data, true_gyro_bias):
        
        # Log pertinent Quantities
        self.log_v(
            "gyro_bias_true.bin",
            [current_time - self.J2000_start_time] + true_gyro_bias.tolist(),
            ["Time [s]"] + self.true_gyro_bias_labels,
        )
        # [TODO]: log measurement data - sun sensor direction
        self.log_v(
            "state_true.bin",
            [current_time - self.J2000_start_time]
            + true_state.tolist()
            + sensor_data.tolist()
            + control_input.tolist(),
            ["Time [s]"] + self.state_labels + self.measurement_labels + self.input_labels,
        )

    def log_estimation(self, current_time, attitude_ekf_state, attitude_estimate_error, gyro_bias_error, EKF_sigmas):

        self.log_v(
            "EKF_state.bin",
            [current_time - self.J2000_start_time] + attitude_ekf_state.tolist(),
            ["Time [s]"] + self.EKF_state_labels,
        )
        self.log_v(
            "EKF_error.bin",
            [current_time - self.J2000_start_time] + attitude_estimate_error.tolist() + gyro_bias_error.tolist(),
            ["Time [s]"] + self.attitude_estimate_error_labels + self.gyro_bias_error_labels,
        )

        self.log_v(
            "state_covariance.bin",
            [current_time - self.J2000_start_time] + EKF_sigmas.tolist(),
            ["Time [s]"] + self.EKF_sigma_labels,
        )