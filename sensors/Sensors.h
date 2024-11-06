#ifndef _SENSOR_
#define _SENSOR_

#include "math/EigenWrapper.h"
#include "ParameterParser.h"

/**
 * @brief Measures the spacecraft position and velocity in ECEF frame
 * 
 * @param pos : position vector in ECI frame [UNITS : m]
 * @param vel : velocity vector in ECI frame [UNITS : m/s]
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return 6 element vector of noisy ECEF positions and velocities
 */
Vector6 GPS(const Vector3 pos, const Vector3 vel, double t_J2000, Simulation_Parameters sc);

/**
 * @brief Measures the % of nominal solar flux incident at each light diode
 * 
 * @param q : Quaternion representing rotation from body frame to ECI
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return N element vector of incident solar fluxes
 */
VectorXd SunSensor(const Vector4 q, double t_J2000, Simulation_Parameters sc);

/**
 * @brief Measures magnetic field reading in the body frame
 * 
 * @param r : Position vector of the satellite in ECI frame [UNTS : m]
 * @param q : Quaternion representing rotation from body frame to ECI
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Magnetic field in the body frame
 */
Vector3 Magnetometer(const Vector3 r, const Vector4 q, double t_J2000, Simulation_Parameters sc);

Vector3 Gyroscope(const Vector3 omega);



/* UTILITY FUNCTIONS */
/**
 * @brief Defines the skew-symmetric form of a matrix
 * 
 * @param v : vector to convert into skew-symmetric form
 * @return 3x3 matrix represnting a skew-symmetric form of a vector
 */
Matrix_3x3 skew_symmetric(Vector3 v);

/**
 * @brief Defines a random rotation about a random axis
 * 
 * @param noise_std : Defines the magnitude of noise [UNITS : rad]
 * @return 3x3 matrix represnting a random rotation
 */
Matrix_3x3 random_SO3_rotation(double noise_std);

#endif