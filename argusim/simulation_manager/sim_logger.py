from argusim.simulation_manager import MultiFileLogger

class SimLogger(MultiFileLogger):
    def __init__(self, log_directory, num_RWs, num_photodiodes, num_MTBs, num_panels, J2000_start_time):
        super().__init__(log_directory)
        self.num_RWs = num_RWs
        self.num_photodiodes = num_photodiodes
        self.num_MTBs = num_MTBs
        self.num_panels = num_panels
        self.J2000_start_time = J2000_start_time

        self.state_labels = ["r_x ECI [m]", 
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
                            "zMag ECI [T]"] + \
                            ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.num_RWs)] + \
                            ["bias_x [deg/s]",
                            "bias_y [deg/s]",
                            "bias_z [deg/s]"]    + \
                            ["Battery SoC", "Battery temperature [K]", "Pack Voltage [V]", "Pack Current [A]"]
            
        self.measurement_labels = [
            "gps_posx ECEF [m]",
            "gps_posy ECEF [m]",
            "gps_posz ECEF [m]",
            "gps_velx ECEF [m/s]",
            "gps_vely ECEF [m/s]",
            "gps_velz ECEF [m/s]",
            "gyro_x [deg/s]",
            "gyro_y [deg/s]",
            "gyro_z [deg/s]",
            "mag_x_body [muT]",
            "mag_y_body [muT]",
            "mag_z_body [muT]"] \
        + ["light_sensor_lux [lx]" + str(i) for i in range(self.num_photodiodes)] \
        + ['mtb_power [W]' + str(i) for i in range(self.num_MTBs)] \
        + ['solar_power [W]' + str(i) for i in range(self.num_panels)] \
        + ["Battery SoC [%]", "Battery Capacity [J]", "Battery Current [A]",
        "Battery Voltage [V]", "Battery Mid Voltage [V]", "Battery TTE [s]",
        "Battery TTF [s]", "Battery Temperature [K]"] + ["Jetson Power [W]"]
        # + ["rw_encoder_" + str(i) + " [rad/s]" for i in range(self.num_RWs)]

        self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.num_MTBs)] \
                          + ["T_RW_" + str(i) + " [Nm]" for i in range(self.num_RWs)] + ["Jetson ON"]

       
    def log_measurements(self, current_time, measurements):
        
        # for now, logging all measurements together since each sensor always outputs measurements 
        self.log_v(
            "measurements.bin",
            [current_time - self.J2000_start_time]
            + measurements.tolist(),
            ["Time [s]"] + self.measurement_labels,
        )
        # [TODO:] log measurements separately for each sensor, only log when there is a new measurement

    def log_true_state(self, current_time, true_state, control_input):
        
        # Log pertinent Quantities
        self.log_v(
            "state_true.bin",
            [current_time - self.J2000_start_time]
            + true_state.tolist()
            + control_input.tolist(),
            ["Time [s]"] + self.state_labels + self.input_labels,
        )