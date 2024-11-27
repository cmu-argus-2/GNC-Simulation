#ifndef C___SRP_H
#define C___SRP_H

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"

/**
 * Determines if the satellite is in sunlight or shadow based on simple cylindrical shadow model.
 * Taken from Montenbruck and Gill p. 80-83
 * @param r_earth position vector of spacecraft with respect to Earth (ECI, meters).
 * @param r_sun position vector of spacecraft with respect to the sun (meters).
 * @return 0.0 if in shadow, 1.0 if in sunlight, 0 to 1.0 if in partial shadow
 */
double partial_illumination(const Vector3& r, const Vector3& r_Sun);

/**
 * Determines if the satellite is in sunlight or shadow based on simple cylindrical shadow model.
 * Taken from Montenbruck and Gill p. 80-83
 * @param r ECI position vector of spacecraft [m].
 * @param r_Sun Sun position vector (geocentric) [m].
 * @return 0.0 if in shadow, 1.0 if in sunlight, 0 to 1.0 if in partial shadow
 */
double partial_illumination_rel(const Vector3& r_earth, const Vector3& r_sun);

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
Vector3 SRP_acceleration(const Quaternion q, const Vector3 r, double t_J2000, double CR, double A, double m);

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