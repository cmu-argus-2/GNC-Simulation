#include "Sensors.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "ParameterParser.h"
#include "SRP.h"
#include "MagneticField.h"
#include <cmath>
#include <random>
#include <iostream>

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif

#define NOMINAL_SOLAR_INTENSITY 1373 // W/m^2

VectorXd ReadSensors(VectorXd &state, const VectorXd control_input, double t_J2000, Simulation_Parameters sc)
{
    /* Measurement Vector: [GPS state          (6x1),
                            IMU reading        (3x1),
                            Lux Readings       (9x1),
                            Power Diagnostics ((6+8)x1)]*/
    int measurement_vec_size = 6 + 6 + sc.num_photodiodes + sc.num_MTBs + 8;
    
    VectorXd measurement = VectorXd::Zero(measurement_vec_size);

    measurement(Eigen::seqN(0,6)) = GPS(state, t_J2000, sc);
    measurement(Eigen::seqN(6,6)) = IMU(state, sc);
    measurement(Eigen::seqN(12, sc.num_photodiodes)) = SunSensor(state, sc);
    measurement(Eigen::seqN(12+sc.num_photodiodes, sc.num_MTBs + 8)) = PowerConsumption(state, control_input, sc);

    return measurement;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ---------------------------------------------------- GPS -------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
Vector6 GPS(const VectorXd &state, double t_J2000, Simulation_Parameters sc)
{
    // Noise Distributions
    static std::normal_distribution<double> pos_noise_dist(0, sc.gps_pos_std);
    static std::normal_distribution<double> vel_noise_dist(0, sc.gps_vel_std);
    
    Vector6 y = Vector6::Zero();

    Matrix_3x3 R_ECI2ECEF = ECI2ECEF(t_J2000);

    Vector3 pos_noise = Vector3::NullaryExpr([&](){return pos_noise_dist(gen);});
    Vector3 vel_noise = Vector3::NullaryExpr([&](){return vel_noise_dist(gen);});

    // GPS returns measurements in ECEF
    y(Eigen::seqN(0,3)) = R_ECI2ECEF*state(Eigen::seqN(0,3)) + sc.gps_pos_std*pos_noise; // Add noise to the measurements
    y(Eigen::seqN(3,3)) = R_ECI2ECEF*state(Eigen::seqN(3,3)) + sc.gps_vel_std*vel_noise;

    return y;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ---------------------------------------------------- IMU -------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd IMU(const VectorXd &state, Simulation_Parameters sc)
{
    VectorXd imu_reading = VectorXd::Zero(6);

    /* Gyroscope */ 
    static Vector3 bias = Vector3::Zero();

    // Gyro Noise Models
    static std::normal_distribution<double> bias_noise_dist(0, sc.gyro_sigma_w*sqrt(sc.dt));
    static std::normal_distribution<double> white_noise_dist(0, sc.gyro_sigma_v/sqrt(sc.dt));

    // Update Bias
    Vector3 bias_noise = Vector3::NullaryExpr([&](){return bias_noise_dist(gen);});
    bias = bias + sc.dt*(bias_noise - bias/sc.gyro_correlation_time);

    // Random white noise
    Vector3 white_noise = Vector3::NullaryExpr([&](){return white_noise_dist(gen);});

    // Noisy Measurement
    Vector3 omega_meas = (1 + sc.gyro_scale_factor_err)*state(Eigen::seqN(10,3)) + bias + white_noise;
    
    imu_reading(Eigen::seqN(0,3)) = omega_meas;

    /* Magnetometer */
    
    // Magnetometer Noise Distribution
    static std::normal_distribution<double> mag_noise_dist(0, sc.magnetometer_noise_std);

    Quaternion quat_BtoECI {state(6), state(7), state(8), state(9)};

    // True Magnetic Field
    Vector3 B_eci = state(Eigen::seqN(16,3));

    // Noisy Measurement
    Vector3 B_body = random_SO3_rotation(mag_noise_dist, gen)*quat_BtoECI.toRotationMatrix().transpose()*B_eci;

    imu_reading(Eigen::seqN(3,3)) = B_body;

    return imu_reading;
}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- LIGHT SENSORS ------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd SunSensor(const VectorXd &state, Simulation_Parameters sc)
{
    // Photodiodes noise distribution
    static std::normal_distribution<double> pd_noise_dist(0, sc.photodiode_std);

    Quaternion quat {state(6), state(7), state(8), state(9)};

    // True sun position
    Vector3 sun_pos_eci = state(Eigen::seqN(13,3));
    Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation

    // Noisy Measurements
    VectorXd photodiode_noise = VectorXd::NullaryExpr(sc.num_photodiodes, [&](){return pd_noise_dist(gen);});
    VectorXd solar_intensity_on_panel = 140000*sc.G_pd_b.transpose()*sun_pos_body/sun_pos_body.norm() + photodiode_noise; // 140,000 : Nominal Solar lux

    solar_intensity_on_panel = (solar_intensity_on_panel.array() < 0.0).select(0, solar_intensity_on_panel); // If the intensity is negative, set to 0

    return solar_intensity_on_panel;

}

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   ------------------------------------------------- POWER CONSUMPTION --------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------------------------------------------- */
VectorXd PowerConsumption(VectorXd &state, const VectorXd control_input, Simulation_Parameters sc)
{
    int reading_size = sc.num_MTBs + 8; // power consumptions for each MTB and 8 battery diagnostics

    VectorXd power_readings = VectorXd::Zero(reading_size);
    
    /* Magnetorquer Power Consumption */ 
    VectorXd mtb_power = Magnetorquers(control_input, sc);
    power_readings(Eigen::seqN(0,sc.num_MTBs)) = mtb_power;

    /* Solar power generation */
    VectorXd solar_power = SolarPanels(state, sc);
    
    /* Static Power Consumption */
    double static_power_draw = sc.mb_power + control_input(sc.num_MTBs+sc.num_RWs)*sc.jetson_power;

    /* Get Battery State */
    double net_power_draw = mtb_power.sum() + static_power_draw - solar_power.sum();
    power_readings(Eigen::seqN(sc.num_MTBs, 8)) = Battery(state, sc, net_power_draw);

    return power_readings;

}

VectorXd Magnetorquers(const VectorXd control_input, Simulation_Parameters sc)
{
    VectorXd power_consumption = control_input(Eigen::seqN(0,sc.num_MTBs)).array() * control_input(Eigen::seqN(0,sc.num_MTBs)).array() / sc.resistances.array();
    return power_consumption;
}

VectorXd SolarPanels(const VectorXd &state, Simulation_Parameters sc)
{
    Quaternion quat {state(6), state(7), state(8), state(9)};

    // True sun position
    Vector3 sun_pos_eci = state(Eigen::seqN(13,3));
    Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation

    // Compute solar power
    VectorXd solar_power = NOMINAL_SOLAR_INTENSITY*sc.G_sp_b.transpose()*sc.solar_panel_efficiency*sc.solar_panel_area*sun_pos_body/sun_pos_body.norm();
    solar_power = (solar_power.array() < 0.0).select(0, solar_power);

    return solar_power;
}

VectorXd Battery(VectorXd &state, Simulation_Parameters sc, double net_power_consumption)
{
    static double net_power_consumed_lpf = 0;

    VectorXd battery_readings = VectorXd::Zero(8);
    
    double power_consumed = net_power_consumption*sc.dt;
    state(19+sc.num_RWs) -= 100*power_consumed/sc.battery_capacity; // Change in SoC
    state(19+sc.num_RWs+3) = std::fabs(power_consumed)/state(19+sc.num_RWs+2); // Current in A
    state(19+sc.num_RWs+1) += (pow(state(19+sc.num_RWs+3),2)*sc.battery_internal_resistance - sc.battery_radiative_loss*pow(state(19+sc.num_RWs+1),4))*sc.dt;

    // Populate battery readings
    net_power_consumed_lpf = net_power_consumed_lpf*0.8 + 0.2*power_consumed;
    battery_readings(0) = state(19+sc.num_RWs);
    battery_readings(1) = sc.battery_capacity;
    battery_readings(2) = state(19+sc.num_RWs+3);
    battery_readings(3) = sc.max_pack_voltage;
    battery_readings(4) = 7.4;
    battery_readings(5) = (net_power_consumed_lpf > 0) ? 0.01*state(19+sc.num_RWs)*sc.battery_capacity/net_power_consumed_lpf : 1.0e10; // TTE
    battery_readings(6) = (net_power_consumed_lpf < 0) ? 0.01*(100-state(19+sc.num_RWs))*sc.battery_capacity/net_power_consumed_lpf : 1.0e10; // TTF
    battery_readings(7) = state(19+sc.num_RWs+1);

    return battery_readings;
}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysensors, m) {
    m.doc() = "pybind11 sensors plugin";   // module docstring    

    m.def("readSensors", &ReadSensors, "Populate Sensor Measurement Vector");
}
#endif