#include "Sensors.h"
#include "math/EigenWrapper.h"
#include "utils_and_transforms.h"
#include "ParameterParser.h"
#include "SRP.h"

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif


Vector6 GPS(const Vector3 pos, const Vector3 vel, double t_J2000, Simulation_Parameters sc)
{
    Vector6 y = Vector6::Zero();

    Matrix_3x3 R_ECI2ECEF = ECI2ECEF(t_J2000);

    Vector3 pos_noise = Vector3::Random();
    Vector3 vel_noise = Vector3::Random();

    // GPS returns measurements in ECEF
    y(Eigen::seqN(0,3)) = R_ECI2ECEF*pos + sc.gps_pos_std*pos_noise; // Add noise to the measurements
    y(Eigen::seqN(3,3)) = R_ECI2ECEF*vel + sc.gps_vel_std*vel_noise;

    return y;
}

VectorXd SunSensor(const Vector4 q, double t_J2000, Simulation_Parameters sc)
{
    Quaternion quat {q(0), q(1), q(2), q(3)};

    Vector3 sun_pos_eci = sun_position_eci(t_J2000);
    Vector3 sun_pos_body = quat.toRotationMatrix().transpose()*sun_pos_eci; // q represents body to ECI transformation

    VectorXd sun_vec_angles = sun_pos_body.transpose()*sc.G_ld_b;

    return sun_vec_angles.cwiseMin(0.0);

}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pysensors, m) {
    m.doc() = "pybind11 sensors plugin";   // module docstring    

    m.def("gps", &GPS, "GPS Sensor");
    m.def("sunSensor", &SunSensor, "Sun Sensor");
}
#endif