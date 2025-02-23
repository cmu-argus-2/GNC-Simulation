import numpy as np
from scipy.spatial.transform import Rotation as R
from argusim.sensors.Sensor import SensorNoiseParams, TriAxisSensor
from argusim.sensors.SunSensor import SunSensor
from argusim.sensors.Magnetometer import Magnetometer

class MeasurementPreprocessing:
    def __init__(self, magnetometer: Magnetometer, sun_sensor: SunSensor, gyro: TriAxisSensor):
        self.magnetometer = magnetometer
        self.sunSensor    = sun_sensor
        self.gyro         = gyro
        self.measurements = {}

    def preprocess_measurements(self, sensor_data, current_time, Idx):
        """
        Preprocess measurements from the sensors in the system.

        Parameters:
        sensor_data (dict): Dictionary containing sensor data with keys 'magnetometer', 'sun_sensor', 'gps', and 'gyrometer'.
        current_time (float): Current time in the simulation.

        Returns:
        dict: Dictionary containing preprocessed sensor data.
        """
        # true_ECI_R_body = R.from_quat([*self.true_state[7:10], self.true_state[6]])

        proc_meas = np.zeros(Idx["NY"])

        self.Idx["Y"] = dict()
        self.Idx["Y"]["GPS_POS"] = slice(0, 3)
        self.Idx["Y"]["GPS_VEL"] = slice(3, 6)
        self.Idx["Y"]["GYRO"] = slice(6, 9)
        self.Idx["Y"]["MAG"] = slice(9, 12)
        self.Idx["Y"]["SUN"] = slice(12, 15) 
        self.Idx["Y"]["RW_OMEGA"] = slice(15, 15+self.num_RWs)

        # Preprocess GPS data
        proc_meas[Idx['Y']["GPS"]], GotGPS = self.preprocess_gps(sensor_data['gps'], current_time)

        # Preprocess gyrometer data
        proc_meas[Idx['Y']["GYRO"]], GotGyro = self.preprocess_gyrometer(sensor_data['gyrometer'], current_time)

        # Preprocess magnetometer data
        proc_meas[Idx['Y']["MAG"]], GotMag = self.preprocess_magnetometer(sensor_data['magnetometer'], current_time)

        # Preprocess sun sensor data
        proc_meas[Idx['Y']["SUN"]], GotSun = self.preprocess_sun_sensor(sensor_data['sun_sensor'], current_time)

        got_flags = {
            "GotGPS": GotGPS,
            "GotGyro": GotGyro,
            "GotMag": GotMag,
            "GotSun": GotSun
        }

        return proc_meas, got_flags

    def preprocess_magnetometer(self, raw_mag_data, cur_time):
        got_B = False
        proc_mag_data = None
        if cur_time >= self.last_magnetometer_measurement_time + self.magnetometer.dt:
            proc_mag_data = np.copy(raw_mag_data)  # Add actual preprocessing logic here
            self.last_magnetometer_measurement_time = cur_time
            got_B = True
        
        return proc_mag_data, got_B

    def preprocess_sun_sensor(self, data, current_time):
        # Sun Sensor update
        got_sun = False
        # if number of active photodiodess > 2, SUN_IN_VIEW = True
        SUN_IN_VIEW = sum(data > self.sunSensor.THRESHOLD_ILLUMINATION_LUX) > 2
        if SUN_IN_VIEW and (current_time >= self.sunSensor.last_meas_time + self.sunSensor.dt):
            
            valid_ids = data > self.sunSensor.THRESHOLD_ILLUMINATION_LUX
            sun_vector = np.linalg.pinv(self.sunSensor.G_pd_b[:,valid_ids]) @ data[valid_ids]
            # direction of photodiodes
            self.sun_sensor.last_meas_time = current_time
            got_sun = True

        return sun_vector, got_sun

    def preprocess_gps(self, data):
        # Implement GPS data preprocessing here
        # Example: Convert coordinates to a standard format
        # TODO simulate RTC and use its drifting time

        return 

    def preprocess_gyrometer(self, data, current_time):
        # Propagate on Gyro
        got_Gyr = False
        if current_time >= self.last_gyro_measurement_time + self.gyro.dt:
            true_omega_body_wrt_ECI_in_body = np.copy(self.true_state[10:13])
            self.measurements[self.Idx["Y"]["GYRO"]] = self.gyro.update(true_omega_body_wrt_ECI_in_body)
            self.last_gyro_measurement_time = current_time

            got_Gyr = True

        return got_Gyr
