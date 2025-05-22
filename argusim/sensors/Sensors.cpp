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
                            Lux Readings       (9x1),
                            solar power        (14x1),  
                            Power Diagnostics ((6+8)x1),
                            Jetson Power       (1x1)]*/
    int measurement_vec_size = 6 + 6 + sc.num_photodiodes + sc.num_MTBs + sc.num_panels + 8 + 1;
    
    VectorXd measurement = VectorXd::Zero(measurement_vec_size);

    measurement(Eigen::seqN(0,6)) = GPS(state, t_J2000, sc);
    measurement(Eigen::seqN(6,6)) = IMU(state, sc);
    measurement(Eigen::seqN(12, sc.num_photodiodes)) = SunSensor(state, sc);

    VectorXd power_readings = PowerReadings(state, control_input, sc);
    measurement(Eigen::seqN(12+sc.num_photodiodes, sc.num_MTBs + sc.num_panels + 8)) = power_readings;
    measurement(12+sc.num_photodiodes + sc.num_MTBs + sc.num_panels + 8) = control_input(sc.num_MTBs+sc.num_RWs)*sc.jetson_power; 

    return measurement;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ---------------------------------------------------- GPS -------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
Vector6 GPS(const VectorXd state, double t_J2000, Simulation_Parameters sc)
{
    // Noise Distributions
    static std::normal_distribution<double> pos_noise_dist(0, sc.gps_pos_std);
    static std::normal_distribution<double> vel_noise_dist(0, sc.gps_vel_std);
    
    Vector6 y = Vector6::Zero();

    Matrix_3x3 R_ECI2ECEF = ECI2ECEF(t_J2000);

    Vector3 pos_noise = Vector3::NullaryExpr([&](){return pos_noise_dist(gen);});
    Vector3 vel_noise = Vector3::NullaryExpr([&](){return vel_noise_dist(gen);});
    
    // GPS returns measurements in ECEF 
    Vector3 OMEGA {0, 0, 7.292115E-5};
    y(Eigen::seqN(0,3)) = R_ECI2ECEF*state(sc.x_idx_map["position"].to_seq()) + sc.gps_pos_std*pos_noise; // Add noise to the measurements

    Vector3 r_ecef = R_ECI2ECEF*state(sc.x_idx_map["position"].to_seq());
    y(Eigen::seqN(3,3)) = R_ECI2ECEF*state(sc.x_idx_map["velocity"].to_seq()) - OMEGA.cross(r_ecef) + sc.gps_vel_std*vel_noise;

    return y;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ---------------------------------------------------- IMU -------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd IMU(const VectorXd state, Simulation_Parameters sc)
{
    VectorXd imu_reading = VectorXd::Zero(9);

    /* Gyroscope */ 

    // Gyro Noise Models
    static std::normal_distribution<double> white_noise_dist(0, sc.gyro_sigma_v);

    // Update Bias 
    Vector3 bias = state(sc.x_idx_map["gyro_bias"].to_seq());

    // Random white noise
    Vector3 white_noise = Vector3::NullaryExpr([&](){return white_noise_dist(gen);});

    // Noisy Measurement 
    Vector3 omega_meas = (1 + sc.gyro_scale_factor_err)*state(sc.x_idx_map["angular_rate"].to_seq()) + bias + white_noise;
    
    imu_reading(Eigen::seqN(0,3)) = omega_meas;

    /* Magnetometer */
    
    // Magnetometer Noise Distribution
    static std::normal_distribution<double> mag_noise_dist(0, sc.magnetometer_noise_std);

    Quaternion quat_BtoECI = state(sc.x_idx_map["quaternion"].to_seq());
    // Quaternion quat_BtoECI {state(6), state(7), state(8), state(9)};
    
    // True Magnetic Field
    Vector3 B_eci = state(sc.x_idx_map["magnetic_field"].to_seq());

    // Noisy Measurement
    Vector3 B_body = random_SO3_rotation(mag_noise_dist, gen)*quat_BtoECI.toRotationMatrix().transpose()*B_eci;

    imu_reading(Eigen::seqN(3,3)) = B_body;

    return imu_reading;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- LIGHT SENSORS ------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc)
{
    // Photodiodes noise distribution
    static std::normal_distribution<double> pd_noise_dist(0, sc.photodiode_std);
    
    Quaternion quat = state(sc.x_idx_map["quaternion"].to_seq());
    // Quaternion quat {state(6), state(7), state(8), state(9)};
    Vector3 r_eci = state(sc.x_idx_map["position"].to_seq());

    // True sun position 
    Vector3 sun_pos_eci = state(sc.x_idx_map["sun_position"].to_seq());
    Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation

    // Shadow Factor
    double shadow = shadow_factor(r_eci, sun_pos_eci);

    // Noisy Measurements
    VectorXd photodiode_noise = VectorXd::NullaryExpr(sc.num_photodiodes, [&](){return pd_noise_dist(gen);});
    VectorXd solar_intensity_on_panel = shadow*140000*sc.G_pd_b.transpose()*sun_pos_body/sun_pos_body.norm() + photodiode_noise; // 140,000 : Nominal Solar lux

    solar_intensity_on_panel = (solar_intensity_on_panel.array() < 0.0).select(0, solar_intensity_on_panel); // If the intensity is negative, set to 0

    return solar_intensity_on_panel;

}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- POWER CONSUMPTION --------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd PowerReadings(const VectorXd state, const VectorXd control_input, Simulation_Parameters sc)
{
    int reading_size = sc.num_MTBs + sc.num_panels + 8; // power consumptions for each MTB and 8 battery diagnostics

    VectorXd power_readings = VectorXd::Zero(reading_size);
    
    /* Magnetorquer Power Consumption */ 
    VectorXd mtb_power = MagnetorquerPower(control_input, sc);
    power_readings(Eigen::seqN(0,sc.num_MTBs)) = mtb_power;

    /* Solar power generation */
    VectorXd solar_power = SolarPanels(state, sc);
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
    battery_readings(0) = state(19+sc.num_RWs);
    battery_readings(1) = sc.battery_capacity;
    battery_readings(2) = state(19+sc.num_RWs+3);
    battery_readings(3) = sc.max_pack_voltage;
    battery_readings(4) = 7.4;
    battery_readings(5) = (state(19+sc.num_RWs+3) < 0) ? 0.01*state(19+sc.num_RWs)*sc.battery_capacity/(-state(19+sc.num_RWs+3)*sc.max_pack_voltage) : 1.0e10; // TTE
    battery_readings(6) = (state(19+sc.num_RWs+3) > 0) ? 0.01*(100-state(19+sc.num_RWs))*sc.battery_capacity/(state(19+sc.num_RWs+3)*sc.max_pack_voltage) : 1.0e10; // TTF
    battery_readings(7) = state(19+sc.num_RWs+1);

    return battery_readings;
}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysensors, m) {
    m.doc() = "pybind11 sensors plugin";   // module docstring 
    m.def("readSensors", &ReadSensors, "Populate Sensor Measurement Vector");
}
#endif