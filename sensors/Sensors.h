#ifndef _SENSOR_
#define _SENSOR_

#include "math/EigenWrapper.h"
#include "ParameterParser.h"
#include <random>

// Random Seed and Algorithm Definition
std::random_device rd;
std::mt19937 gen(rd());

/**
 * @brief Measures the spacecraft position and velocity in ECEF frame
 * 
 * @param state : true state vector
 * @param t_J2000 : seconds since J2000 used to read sensors
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return measurement vector with added noise
 */
VectorXd ReadSensors(const VectorXd state, double t_J2000, Simulation_Parameters sc);

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
 * @param state : True state evctor of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return N element vector of incident solar fluxes
 */
VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures magnetic field reading in the body frame
 * 
 * @param state : True state evctor of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Magnetic field in the body frame
 */
Vector3 Magnetometer(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures magnetic field reading in the body frame
 * 
 * @param omega : True angular rate of the satellite [UNTS : rad/s]
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Measured angular rate through the gyroscope
 */
Vector3 Gyroscope(const Vector3 omega, Simulation_Parameters sc);



/* UTILITY FUNCTIONS */

/**
 * @brief Defines a random rotation about a random axis
 * 
 * @param dist : Normal Distribution characterizing the noise profile
 * @return 3x3 matrix represnting a random rotation
 */
Matrix_3x3 random_SO3_rotation(std::normal_distribution<double> dist);

#endif