#ifndef _SENSOR_
#define _SENSOR_

#include "math/EigenWrapper.h"
#include "ParameterParser.h"
#include <random>

// Random Seed and Algorithm Definition
std::random_device rd;
std::mt19937 gen(rd());

/**
 * @brief Populates the measurement vector by querying measurements from each sensor
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
 * @param state : True state vector of the satellite
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return 6 element vector of noisy ECEF positions and velocities
 */
Vector6 GPS(const VectorXd state, double t_J2000, Simulation_Parameters sc);

/**
 * @brief Measures the % of nominal solar flux incident at each light diode
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return N element vector of incident solar fluxes
 */
VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures magnetic field reading in the body frame
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Magnetic field in the body frame
 */
Vector3 Magnetometer(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures Angular Velocity reading in the body frame
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Measured angular rate through the gyroscope
 */
Vector3 Gyroscope(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures Reaction Wheel Encoder readings
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Measured reaction wheel encoder readings
 */
VectorXd RWEncoder(const VectorXd state, Simulation_Parameters sc);

#endif