#include "Sensors.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "ParameterParser.h"
#include "SRP.h"
#include "MagneticField.h"
#include "power.h"
#include <cmath>
#include <functional>
#include <random>
#include <iostream>

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif

#define NOMINAL_SOLAR_INTENSITY 1373 // W/m^2

VectorXd ReadSensors(const VectorXd state, const VectorXd control_input, double t_J2000, Simulation_Parameters sc)
{
    /* Measurement Vector: [GPS state          (6x1),
                            IMU reading        (6x1),
                            Star Tracker       (4x1), (optional)
                            Lux Readings       (9x1),
                            solar power        (14x1),  
                            Power Diagnostics ((6+8)x1),
                            Jetson Power       (1x1)]*/
    int measurement_vec_size = 6 + 6 + sc.num_stk + sc.num_photodiodes + sc.num_MTBs + sc.num_panels + 8 + 1;

    VectorXd measurement = VectorXd::Zero(measurement_vec_size);

    measurement(sc.y_idx_map["gps"].to_seq()) = GPS(state, t_J2000, sc);

    measurement(sc.y_idx_map["imu"].to_seq()) = IMU(state, sc);

    // If including star tracker, populate the star tracker measurement
    if (sc.include_star_tracker) {
        measurement(sc.y_idx_map["star_tracker"].to_seq()) = StarTracker(state, sc);
    }

    measurement(sc.y_idx_map["photodiode"].to_seq()) = SunSensor(state, sc);

    VectorXd power_readings = PowerReadings(state, control_input, sc);

    measurement(sc.y_idx_map["power_readings"].to_seq()) = power_readings;

    measurement(sc.y_idx_map["jetson_power"].to_idx()) = control_input(sc.u_idx_map["jet_power"].to_idx())*sc.jetson_power;

    return measurement;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ---------------------------------------------------- GPS -------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
Vector6 GPS(const VectorXd state, double t_J2000, Simulation_Parameters sc)
{
    Vector6 y = Vector6::Zero();
    /*
    Matrix_3x3 R_ECI2ECEF = ECI2ECEF(t_J2000);
    
    // GPS returns measurements in ECEF 
    Vector3 OMEGA {0, 0, 7.292115E-5};

    y(Eigen::seqN(0,3)) = R_ECI2ECEF*state(sc.x_idx_map["position"].to_seq()) + pos_noise; // Add noise to the measurements

    Vector3 r_ecef = R_ECI2ECEF*state(sc.x_idx_map["position"].to_seq());
    y(Eigen::seqN(3,3)) = R_ECI2ECEF*state(sc.x_idx_map["velocity"].to_seq()) - OMEGA.cross(r_ecef) + vel_noise;
    */
    Matrix_6x6 R_ECI2ECEF = ECI2ECEF_rv(t_J2000);
    y(Eigen::seqN(0,6)) = R_ECI2ECEF * state(sc.x_idx_map["translation"].to_seq());

    if (sc.perfect_sensors) {
        return y;
    }

    // Noise Distributions
    static std::normal_distribution<double> pos_noise_dist(0, sc.gps_pos_std);
    static std::normal_distribution<double> vel_noise_dist(0, sc.gps_vel_std);

    Vector3 pos_noise = Vector3::NullaryExpr([&](){return pos_noise_dist(gen);});
    Vector3 vel_noise = Vector3::NullaryExpr([&](){return vel_noise_dist(gen);});

    y(Eigen::seqN(0,3)) += pos_noise;
    y(Eigen::seqN(3,3)) += vel_noise;
    return y;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ---------------------------------------------------- IMU -------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
Vector3 Gyroscope(const VectorXd state, Simulation_Parameters sc)
{
    /* Gyroscope */ 
    Vector3 omega_true = state(sc.x_idx_map["angular_rate"].to_seq()) * (180.0 / M_PI); // rad/s to deg/s

    if (sc.perfect_sensors) {
        return omega_true;
    }

    // Gyro Noise Models
    static std::normal_distribution<double> white_noise_dist(0, sc.gyro_sigma_v);

    // Update Bias 
    Vector3 bias = state(sc.x_idx_map["gyro_bias"].to_seq());

    // Random white noise
    Vector3 white_noise = Vector3::NullaryExpr([&](){return white_noise_dist(gen);});

    // Noisy Measurement 
    Vector3 omega_meas = (1 + sc.gyro_scale_factor_err)*omega_true + bias + white_noise;
    // Enforce gyro range limits
    for (int i = 0; i < 3; ++i) {
        if (omega_meas(i) > sc.gyro_range) omega_meas(i) = sc.gyro_range;
        else if (omega_meas(i) < -sc.gyro_range) omega_meas(i) = -sc.gyro_range;
    }
    // Quantize the gyro measurement
    omega_meas = (omega_meas.array()/sc.gyro_resolution).round()*  sc.gyro_resolution; // Round to the nearest resolution
    return omega_meas;
}


Vector3 Magnetometer(const VectorXd state, Simulation_Parameters sc)
{
    Quaternion quat_BtoECI = vectorToQuaternion(state(sc.x_idx_map["quaternion"].to_seq()));

    // True Magnetic Field
    Vector3 B_eci = state(sc.x_idx_map["magnetic_field"].to_seq()) * 1e6; // Convert from T to uT

    // Magnetic field in the body frame
    Vector3 B_body = quat_BtoECI.toRotationMatrix().transpose() * B_eci;

    if (sc.perfect_sensors) {
        return B_body;
    }

    // Magnetometer Noise Distribution
    static std::normal_distribution<double> mag_noise_dist(0, sc.magnetometer_noise_std);

    // Noisy Measurement
    // Vector3 B_body_meas = random_SO3_rotation(mag_noise_dist, gen) * B_body;
    Vector3 B_body_meas = B_body + Vector3::NullaryExpr([&](){return mag_noise_dist(gen);});

    // Effect of magnetorquers on the magnetic field
    VectorXd mtb_currents = state(sc.x_idx_map["mtb_currents"].to_seq());
    Vector3 mtb_B_effect = sc.MTB.getMagneticFieldAtMagnetometer(mtb_currents);
    B_body_meas += mtb_B_effect * 1e6;

    // Enforce magnetometer range limits
    for (int i = 0; i < 2; ++i) {
        if (B_body_meas(i) > sc.magnetometer_range_xy) B_body_meas(i) = sc.magnetometer_range_xy;
        else if (B_body_meas(i) < -sc.magnetometer_range_xy) B_body_meas(i) = -sc.magnetometer_range_xy;
    }
    if (B_body_meas(2) > sc.magnetometer_range_z) B_body_meas(2) = sc.magnetometer_range_z;
    else if (B_body_meas(2) < -sc.magnetometer_range_z) B_body_meas(2) = -sc.magnetometer_range_z;
    
    // Quantize the magnetometer measurement
    B_body_meas = (B_body_meas.array()/sc.magnetometer_resolution).round() * sc.magnetometer_resolution;

    return B_body_meas;
}

VectorXd IMU(const VectorXd state, Simulation_Parameters sc)
{
    VectorXd imu_reading = VectorXd::Zero(6);

    /* Gyroscope */ 
    imu_reading(Eigen::seqN(0,3)) = Gyroscope(state, sc);

    /* Magnetometer */
    imu_reading(Eigen::seqN(3,3)) = Magnetometer(state, sc);

    return imu_reading;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------ STAR TRACKER --------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
Vector4 StarTracker(const VectorXd state, Simulation_Parameters sc)
{
    // Quaternion representing the body frame to ECI frame transformation
    Quaternion quat = vectorToQuaternion(state(sc.x_idx_map["quaternion"].to_seq()));

    if (sc.perfect_sensors) {
        Vector4 exact_quat{quat.w(), quat.x(), quat.y(), quat.z()};
        return exact_quat;
    }

    // Star Tracker Noise Distribution
    static std::normal_distribution<double> star_tracker_noise_dist(0, sc.star_tracker_std);

    Quaternion noise_quat(random_SO3_rotation(star_tracker_noise_dist, gen));
    // Apply noise to the quaternion
    Quaternion noisy_quat = quat * noise_quat;
    Vector4 star_tracker_measurement{noisy_quat.w(), noisy_quat.x(), noisy_quat.y(), noisy_quat.z()};

    return star_tracker_measurement;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- LIGHT SENSORS ------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc)
{
    Quaternion quat = vectorToQuaternion(state(sc.x_idx_map["quaternion"].to_seq()));
    // Quaternion quat {state(6), state(7), state(8), state(9)};
    Vector3 r_eci = state(sc.x_idx_map["position"].to_seq());

    // True sun position 
    Vector3 sun_pos_eci = state(sc.x_idx_map["sun_position"].to_seq());
    Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation

    // Shadow Factor
    double shadow = shadow_factor(r_eci, sun_pos_eci);

    VectorXd solar_intensity_on_panel = shadow*140000*sc.G_pd_b.transpose()*sun_pos_body/sun_pos_body.norm();

    if (sc.perfect_sensors) {
        solar_intensity_on_panel = (solar_intensity_on_panel.array() < 0.0).select(0, solar_intensity_on_panel);
        return solar_intensity_on_panel;
    }

    // Photodiodes noise distribution
    static std::normal_distribution<double> pd_noise_dist(0, sc.photodiode_std);

    // Noisy Measurements
    VectorXd photodiode_noise = VectorXd::NullaryExpr(sc.num_photodiodes, [&](){return pd_noise_dist(gen);});
    solar_intensity_on_panel += photodiode_noise; // 140,000 : Nominal Solar lux

    solar_intensity_on_panel = (solar_intensity_on_panel.array() < 0.0).select(0, solar_intensity_on_panel); // If the intensity is negative, set to 0

    return solar_intensity_on_panel;

}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- POWER CONSUMPTION --------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd PowerReadings(const VectorXd state, const VectorXd control_input, Simulation_Parameters sc)
{
    int reading_size = sc.y_idx_map["power_readings"].get_length(); // power consumptions for each MTB and 8 battery diagnostics
    
    VectorXd power_readings = VectorXd::Zero(reading_size);
    
    /* Magnetorquer Power Consumption */ 
    VectorXd mtb_currents = state(sc.x_idx_map["mtb_currents"].to_seq());
    VectorXd mtb_power = MagnetorquerPower(control_input, mtb_currents, sc.u_idx_map);
    power_readings(Eigen::seqN(0,sc.num_MTBs)) = mtb_power;

    /* Solar power generation */
    VectorXd solar_power = SolarPanels(state, sc.G_sp_b, sc.solar_panel_efficiency, sc.solar_panel_area, sc.x_idx_map);
    power_readings(Eigen::seqN(sc.num_MTBs, sc.num_panels)) = solar_power;
    
    /* Get Battery State */
    VectorXd battery_readings = BatteryReadings(state, sc);
    power_readings(Eigen::seqN(sc.num_MTBs + sc.num_panels, 8)) = battery_readings;

    return power_readings;
}

VectorXd BatteryReadings(const VectorXd state, Simulation_Parameters sc)
{
    VectorXd battery_readings = VectorXd::Zero(8);
   
    // Populate battery readings
    int idx_bat_soc  = sc.x_idx_map["battery_soc"].to_idx();
    int idx_bat_temp = sc.x_idx_map["battery_temp"].to_idx();
    // int idx_bat_volt = sc.x_idx_map["battery_voltage"].to_idx();
    int idx_bat_cur  = sc.x_idx_map["battery_current"].to_idx();

    battery_readings(0) = state(idx_bat_soc);
    battery_readings(1) = sc.battery_capacity;
    battery_readings(2) = state(idx_bat_cur);
    battery_readings(3) = sc.max_pack_voltage;
    battery_readings(4) = 7.4;
    battery_readings(5) = (state(idx_bat_cur) < 0) ? 0.01*state(idx_bat_soc)*sc.battery_capacity/(-state(idx_bat_cur)*sc.max_pack_voltage) : 1.0e10; // TTE
    battery_readings(6) = (state(idx_bat_cur) > 0) ? 0.01*(100-state(idx_bat_soc))*sc.battery_capacity/(state(idx_bat_cur)*sc.max_pack_voltage) : 1.0e10; // TTF
    battery_readings(7) = state(idx_bat_temp);

    return battery_readings;
}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysensors, m) {
    m.doc() = "pybind11 sensors plugin";   // module docstring 
    m.def("readSensors", &ReadSensors, "Populate Sensor Measurement Vector");
}
#endif