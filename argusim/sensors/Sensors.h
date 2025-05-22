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
 * @param control_input : control input
 * @param t_J2000 : seconds since J2000 used to read sensors
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return measurement vector with added noise
 */
VectorXd ReadSensors(const VectorXd state, const VectorXd control_input, double t_J2000, Simulation_Parameters sc);

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
 * @brief Measures Angular Velocity and local magnetic field readings in the body frame
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return Measured IMU reading
 */
 VectorXd IMU(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures the % of nominal solar flux incident at each light diode
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return N element vector of incident solar fluxes
 */
VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Returns the power consumption diagnostics
 * 
 * @param state : true state vector
 * @param control_input : control input
 * @param t_J2000 : seconds since J2000 used to read sensors
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return power consumption measurements
 */
 VectorXd PowerReadings(const VectorXd state, const VectorXd control_input, Simulation_Parameters sc);

/**
 * @brief Measures the power consumed by each actuator
 * 
 * @param state : true state vector
 * @param sc : Instance of ParameterParser class holding Sensor noise characterizations
 * @return battery diagnostics information
 */
VectorXd BatteryReadings(const VectorXd state, Simulation_Parameters sc);

#endif