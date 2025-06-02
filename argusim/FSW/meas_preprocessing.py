import numpy as np
from scipy.spatial.transform import Rotation as R
from argusim.sensors.Sensor import SensorNoiseParams, TriAxisSensor
from argusim.sensors.SunSensor import SunSensor
from argusim.sensors.Magnetometer import Magnetometer
from argusim.sensors.GPS import GPS
from argusim.FSW.fsw_utils import eci_to_ecef, j2000_to_unix_time
class MeasurementPreprocessing:
    def __init__(self, magnetometer: Magnetometer, sun_sensor: SunSensor, gyro: TriAxisSensor, gps: GPS, num_RWs: int):
        self.magnetometer = magnetometer
        self.sunSensor    = sun_sensor
        self.gyro         = gyro
        self.gps          = gps
        # unprocessed, sensor data idx
        num_photodiodes = self.sunSensor.num_photodiodes
        self.Idx = {} 
        self.Idx["NY"] = 12+num_RWs+num_photodiodes
        self.Idx["Y"] = dict()
        self.Idx["Y"]["GPS"] = slice(0, 6)
        self.Idx["Y"]["GPS_POS"] = slice(0, 3)
        self.Idx["Y"]["GPS_VEL"] = slice(3, 6)
        self.Idx["Y"]["GYRO"] = slice(6, 9)
        self.Idx["Y"]["MAG"] = slice(9, 12)
        self.Idx["Y"]["SUN"] = slice(12, 12+num_photodiodes) # self.sunSensor.num_photodiodes
        self.Idx["Y"]["RW_OMEGA"] = slice(12+num_photodiodes, 12+num_photodiodes+num_RWs)          

    def preprocess_measurements(self, sensor_data, current_time, Idxp):
        """
        Preprocess measurements from the sensors in the system.

        Parameters:
        sensor_data (dict): Dictionary containing sensor data with keys 'magnetometer', 'sun_sensor', 'gps', and 'gyrometer'.
        current_time (float): Current time in the simulation.

        Returns:
        dict: Dictionary containing preprocessed sensor data.
        """
        # [TODO:] whether there are new measurements or not should be checked in or before the C++ code to avoid 
        # computing meas values every time step

        proc_meas = np.zeros(Idxp["NY"])

        # Preprocess GPS data
        proc_meas[Idxp['Y']["GPS"]], GotGPS = self.preprocess_gps(sensor_data[self.Idx["Y"]["GPS"]], current_time)

        # Preprocess gyrometer data
        proc_meas[Idxp['Y']["GYRO"]], GotGyro = self.preprocess_gyrometer(sensor_data[self.Idx["Y"]["GYRO"]], current_time)

        # Preprocess magnetometer data
        proc_meas[Idxp['Y']["MAG"]], GotMag = self.preprocess_magnetometer(sensor_data[self.Idx["Y"]["MAG"]], current_time)

        # Preprocess sun sensor data
        proc_meas[Idxp['Y']["SUN"]], GotSun = self.preprocess_sun_sensor(sensor_data[self.Idx["Y"]["SUN"]], current_time)

        # Preprocess reaction wheel encoder data 
        # [TODO]: add preprocessing function
        proc_meas[Idxp['Y']["RW_OMEGA"]] = sensor_data[self.Idx["Y"]["RW_OMEGA"]]

        got_flags = {
            "GotGPS": GotGPS,
            "GotGyro": GotGyro,
            "GotMag": GotMag,
            "GotSun": GotSun,
            "GotRW": True
        }

        return proc_meas, got_flags

    def preprocess_magnetometer(self, raw_mag_data, cur_time):
        got_B = False
        proc_mag_data = None
        if cur_time >= self.magnetometer.last_meas_time + self.magnetometer.dt:
            proc_mag_data = np.copy(raw_mag_data)  # Add actual preprocessing logic here
            self.magnetometer.last_meas_time = cur_time
            self.magnetometer.last_measurement = proc_mag_data
            got_B = True
        
        return self.magnetometer.last_measurement, got_B

    def preprocess_sun_sensor(self, data, current_time):
        # Sun Sensor update
        got_sun = False
        # if number of active photodiodess > 2, SUN_IN_VIEW = True
        SUN_IN_VIEW = sum(data > self.sunSensor.THRESHOLD_ILLUMINATION_LUX) > 2
        if SUN_IN_VIEW and (current_time >= self.sunSensor.last_meas_time + self.sunSensor.dt):
            
            valid_ids = data > self.sunSensor.THRESHOLD_ILLUMINATION_LUX
            sun_vector = np.linalg.pinv(self.sunSensor.G_pd_b[valid_ids,:]) @ data[valid_ids]
            sun_vector /= np.linalg.norm(sun_vector)
            # direction of photodiodes
            self.sunSensor.last_meas_time = current_time
            self.sunSensor.last_measurement = sun_vector
            got_sun = True

        return self.sunSensor.last_measurement, got_sun

    def preprocess_gps(self, data, current_time):
        # Implement GPS data preprocessing here
        # Example: Convert coordinates to a standard format
        # TODO simulate RTC and use its drifting time
        GotGPS = False
        if (current_time >= self.gps.last_meas_time + self.gps.dt):
            self.gps.last_meas_time = current_time
            # convert from ECEF to ECI
            unix_timestamp = j2000_to_unix_time(current_time)
            ecef_eci = eci_to_ecef(unix_timestamp)
            eci_ecef = ecef_eci.transpose()
            data[:3] = eci_ecef @ data[:3]
            data[3:] = eci_ecef @ data[3:]
            self.gps.last_measurement = data
            GotGPS = True

        return self.gps.last_measurement, GotGPS

    def preprocess_gyrometer(self, data, current_time):
        # Propagate on Gyro
        got_Gyr = False
        if current_time >= self.gyro.last_meas_time + self.gyro.dt:
            gyro_meas = np.copy(data)
            self.gyro.last_meas_time = current_time
            self.gyro.last_measurement = gyro_meas
            got_Gyr = True

        return self.gyro.last_measurement, got_Gyr
