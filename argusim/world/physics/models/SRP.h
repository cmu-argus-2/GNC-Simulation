#ifndef C___SRP_H
#define C___SRP_H

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"


double shadow_factor(const Vector3 r_earth, const Vector3 r_sun);

/**
 * @brief Compute SRP force on satellite in J2000 ECI frame
 * 
 * @param q : quaternion vector representing rotation from body frame to ECI
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @param CR : satellite Coefficient of reflectivity 
 * @param A : satellite face area [m]
 * @param m : satellite mass [kg]
 * @return acceleration due to solar radiation pressure [m/s^2]
 */
Vector3 SRP_acceleration(const Vector3 r, const Quaternion q, double t_J2000, double CR, double A, double m);

/**
 * @brief Computes area perpendicular to the velocity vector as a % of face area
 * 
 * @param q : Quaternion representing rotation from body frame to ECI
 * @param r : position vector along which to compute frontal area
 * @return projection_factor where area perpendicular to sun position vector = projection_factor*A_face
 */
double FrontalAreaFactor(const Quaternion q, const Vector3 r);

/**
 * @brief Compute sun position in J2000 ECI frame
 *
 * @param t_J2000 : seconds since the J2000 epoch
 * @return Sun position vector in ECI frame
 */
Vector3 sun_position_eci(double t_J2000);


#endif   // C___SRP_H