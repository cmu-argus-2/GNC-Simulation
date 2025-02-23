#include "Sensors.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "ParameterParser.h"
#include "SRP.h"
#include "MagneticField.h"
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

const double MAX_RANGE = 117000;  // OPT4001
const double THRESHOLD_ILLUMINATION_LUX = 3000;

VectorXd ReadSensors(const VectorXd state, double t_J2000, Simulation_Parameters sc)
{
    gen.seed(std::hash<double>{}(t_J2000)); // set seed for determinism
    int measurement_vec_size = 6 + 3 + 3 + sc.num_photodiodes; // GPS + Gyro + Mag Field + Light Sensors
    VectorXd measurement = VectorXd::Zero(measurement_vec_size);

    measurement(Eigen::seqN(0,6)) = GPS(state, t_J2000, sc);
    measurement(Eigen::seqN(6,3)) = Gyroscope(state, sc);
    measurement(Eigen::seqN(9,3)) = Magnetometer(state, sc);
    measurement(Eigen::seqN(12, sc.num_photodiodes)) = SunSensor(state, sc);

    return measurement;
}

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
    y(Eigen::seqN(0,3)) = R_ECI2ECEF*state(Eigen::seqN(0,3)) + sc.gps_pos_std*pos_noise; // Add noise to the measurements
    y(Eigen::seqN(3,3)) = R_ECI2ECEF*state(Eigen::seqN(3,3)) + sc.gps_vel_std*vel_noise;

    return y;
}


VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc)
{
    // Photodiodes noise distribution
    static std::normal_distribution<double> pd_noise_dist(0, sc.photodiode_std);

    Quaternion quat {state(6), state(7), state(8), state(9)};

    // True sun position
    Vector3 sun_pos_eci = state(Eigen::seqN(13,3));
    Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation

    // Noisy Measurements
    VectorXd photodiode_noise = VectorXd::NullaryExpr(sc.num_photodiodes, [&](){return pd_noise_dist(gen);});
    VectorXd solar_intensity_on_panel = MAX_RANGE*sc.G_pd_b.transpose()*sun_pos_body/sun_pos_body.norm() + photodiode_noise; // 140,000 : Nominal Solar lux

    solar_intensity_on_panel = (solar_intensity_on_panel.array() < 0.0).select(0, solar_intensity_on_panel); // If the intnesity is negative, set to 0

    double shadow_factor = partial_illumination(state(Eigen::seqN(0,3)), sun_pos_eci);

    solar_intensity_on_panel *= shadow_factor;
    
    return solar_intensity_on_panel;

}

Vector3 Magnetometer(const VectorXd state, Simulation_Parameters sc)
{
    // Magnetometer Direction Noise Distribution
    static std::normal_distribution<double> mag_dir_noise_dist(0, sc.sigma_magnetometer);

    Quaternion quat_BtoECI {state(6), state(7), state(8), state(9)};

    // True Magnetic Field
    Vector3 B_eci = state(Eigen::seqN(16,3));

    // Noisy Measurement
    Vector3 B_body = random_SO3_rotation(mag_dir_noise_dist, gen)*quat_BtoECI.toRotationMatrix().transpose()*B_eci;

    return B_body;
}

Vector3 Gyroscope(const VectorXd state, Simulation_Parameters sc)
{
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
    
    return omega_meas;
}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysensors, m) {
    m.doc() = "pybind11 sensors plugin";   // module docstring    

    m.def("readSensors", &ReadSensors, "Populate Sensor Measurement Vector");
    m.def("readGPS", &GPS, "Populate GPS Measurement Vector");
    m.def("readGyroscope", &Gyroscope, "Populate Gyroscope Measurement Vector");
    m.def("readSunSensor", &SunSensor, "Populate Sun Sensor Measurement Vector");
    m.def("readMagnetometer", &Magnetometer, "Populate Magnetometer Measurement Vector");
}
#endif