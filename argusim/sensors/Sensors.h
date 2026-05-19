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
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return measurement vector with added noise
 */
VectorXd ReadSensors(const VectorXd state, const VectorXd control_input, double t_J2000, Simulation_Parameters sc);

/**
 * @brief Measures the spacecraft position and velocity in ECEF frame using the GPS
 * 
 * @param state : True state vector of the satellite
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return 6 element vector of noisy ECEF positions and velocities
 */
Vector6 GPS(const VectorXd state, double t_J2000, Simulation_Parameters sc);

/**
 * @brief Measures the local magnetic field in the body frame using the magnetometer
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return Measured magnetic field vector in body frame
 */
Vector3 Magnetometer(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures the angular velocity in the body frame using the gyroscope
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return Measured angular velocity vector in body frame
 */
Vector3 Gyroscope(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures Angular Velocity and local magnetic field readings in the body frame
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return Measured IMU reading
 */
 VectorXd IMU(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures the attitude quaternion using the star tracker
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return Measured attitude quaternion
 */
 Vector4 StarTracker(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Measures the nominal solar flux incident at each light diode
 * 
 * @param state : True state vector of the satellite
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return N element vector of incident solar fluxes
 */
VectorXd SunSensor(const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Gets RTC time with drift and given resolution
 * 
 * @param t_J2000 : seconds since J2000 used to read sensors
 * @param state : true state vector
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return measured RTC time
 */
double RTC(double t_J2000, const VectorXd state, Simulation_Parameters sc);

/**
 * @brief Returns the power consumption/generation diagnostics. This includes:
 * 1 - Power consumption of each magnetorquer [W]
 * 2 - Power generation of each solar panel [W]
 * 3 - Battery diagnostics
 * 
 * @param state : true state vector
 * @param control_input : control input
 * @param t_J2000 : seconds since J2000 used to read sensors
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return power consumption measurements
 */
 VectorXd PowerReadings(const VectorXd state, const VectorXd control_input, Simulation_Parameters sc);

/**
 * @brief Computes the Battery diagnostics information. This includes: 
 * 1 - State of Charge (SOC) [%]
 * 2 - Battery Capacity [Ah]
 * 3 - Battery Current [A]
 * 4 - Maximum Pack Voltage [V]
 * 5 - Nominal Voltage [V]
 * 6 - Time to Empty (TTE) [s]
 * 7 - Time to Full (TTF) [s]
 * 8 - Battery Temperature [K]
 * 9 - Battery Temperature AIN1 [K]
 * 10 - Battery Temperature AIN2 [K]
 * 11 - Battery Die Temperature [K]
 * 
 * @param state : true state vector
 * @param sc : Instance of Simulation_Parameters class holding Sensor noise characterizations
 * @return battery diagnostics information
 */
VectorXd BatteryReadings(const VectorXd state, Simulation_Parameters sc);

#endif