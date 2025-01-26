#ifndef C___drag_H
#define C___drag_H

#include "math/EigenWrapper.h"

/**
 * @brief Compute drag force on satellite in J2000 ECI frame
 * 
 * @param r : position vector in ECI frame [UNITS : m]
 * @param v : velocity vector in ECI frame [UNITS : m/s]
 * @param q : quaternion vector representing rotation from body frame to ECI
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param Cd : satellite Cd 
 * @param A : satellite face area [m]
 * @param m : satellite mass [kg]
 * @return drag acceleration in ECI frame [m/s^2]
 */
Vector3 drag_acceleration(const Vector3 r, const Vector3 v, const Quaternion q, double t_J2000, double Cd, double A, double m);

/**
 * @brief Compute drag torque on satellite in body frame
 * 
 * This function calculates the torque due to aerodynamic drag acting on the satellite.
 * The drag torque is computed based on the satellite's orientation, velocity, and 
 * atmospheric conditions. The resulting torque is expressed in the body frame.
 * 
 * @return drag torque in body frame [Nm]
 */
Vector3 drag_torque();

/**
 * @brief Compute atmospheric density at current time
 * 
 * @param r : position vector in ECI frame [UNITS : m]
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @return atmospheric density (kg/m^3) at current position
 */
double density(const Vector3 r, double t_J2000);

#endif   // C___drag_H