from argusim.simulation_manager import MultiFileLogger
from argusim.simulation_manager.indexing import IDX
class SimLogger(MultiFileLogger):
    def __init__(self, log_directory, Idx: IDX, J2000_start_time):
        super().__init__(log_directory)
        self.Idx = Idx
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
                            ["I_MTB_" + str(i) + " [A]" for i in range(self.Idx.NMTBS)] + \
                            ["omega_RW_" + str(i) + " [rad/s]" for i in range(self.Idx.NRWS)] + \
                            ["bias_x [deg/s]",
                            "bias_y [deg/s]",
                            "bias_z [deg/s]"]    + \
                            ["rtc_bias [s]"] + \
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
            "mag_z_body [muT]"]
        if self.Idx.NSTK > 0:
            self.measurement_labels += ["star_tracker_qw [-]", "star_tracker_qx [-]", 
                                        "star_tracker_qy [-]", "star_tracker_qz [-]"]  # self.Idx.NPHOTODIODES
        
        self.measurement_labels += ["light_sensor_lux [lx]" + str(i) for i in range(self.Idx.NPHOTODIODES)] \
                                + ["RTC_time [s]"] \
                                + ['mtb_power [W]' + str(i) for i in range(self.Idx.NMTBS)] \
                                + ['solar_power [W]' + str(i) for i in range(self.Idx.NPANELS)] \
                                + ["Battery SoC [%]", "Battery Capacity [J]", "Battery Current [A]",
                                "Battery Voltage [V]", "Battery Mid Voltage [V]", "Battery TTE [s]",
                                "Battery TTF [s]", "Battery Temperature [K]"] + ["Jetson Power [W]"]
                                # + ["rw_encoder_" + str(i) + " [rad/s]" for i in range(self.num_RWs)]
        if self.Idx.NDEPLOYS > 0:
            self.measurement_labels += ["deployment_sensor [mm]" + str(i) for i in range(self.Idx.NDEPLOYS)]

        self.fsw_labels = ["fsw_gps_posx ECI [m]", 
                            "fsw_gps_posy ECI [m]", 
                            "fsw_gps_posz ECI [m]", 
                            "fsw_gps_velx ECI [m/s]", 
                            "fsw_gps_vely ECI [m/s]", 
                            "fsw_gps_velz ECI [m/s]",
                            "fsw_qw",
                            "fsw_qx",
                            "fsw_qy",
                            "fsw_qz",
                            "fsw_gyro_x [rad/s]", 
                            "fsw_gyro_y [rad/s]", 
                            "fsw_gyro_z [rad/s]",
                            "fsw_bias_x [rad/s]",
                            "fsw_bias_y [rad/s]",
                            "fsw_bias_z [rad/s]", 
                            "fsw_mag_x_body [T]", 
                            "fsw_mag_y_body [T]", 
                            "fsw_mag_z_body [T]",
                            "fsw_sun_x",
                            "fsw_sun_y",
                            "fsw_sun_z",
                            "fsw_sun_eci_x",
                            "fsw_sun_eci_y",
                            "fsw_sun_eci_z",
                            "fsw_mag_eci_x",
                            "fsw_mag_eci_y",
                            "fsw_mag_eci_z"]

        self.input_labels = ["V_MTB_" + str(i) + " [V]" for i in range(self.Idx.NMTBS)] \
                          + ["T_RW_" + str(i) + " [Nm]" for i in range(self.Idx.NRWS)] + ["Jetson ON"]

       
    def log_measurements(self, current_time, measurements):
        
        # for now, logging all measurements together since each sensor always outputs measurements 
        self.log_v(
            "measurements.bin",
            [current_time - self.J2000_start_time]
            + measurements.tolist(),
            ["Time [s]"] + self.measurement_labels,
        )

    def log_true_state(self, current_time, true_state, control_input):
        
        # Log pertinent Quantities
        self.log_v(
            "state_true.bin",
            [current_time - self.J2000_start_time]
            + true_state.tolist()
            + control_input.tolist(),
            ["Time [s]"] + self.state_labels + self.input_labels,
        )

    def log_fsw_state(self, current_time, fsw_state, control_input):
        # Log pertinent Quantities
        self.log_v(
            "state_fsw.bin",
            [current_time - self.J2000_start_time]
            + fsw_state.tolist()
            + control_input.tolist(),
            ["Time [s]"] + self.fsw_labels + self.input_labels,
        )