#ifndef C___gravity_H
#define C___gravity_H

#include "math/EigenWrapper.h"

/* CONSTANTS */
double mu = 3.98600435507e14;
double J2 = 1.08262668e-3;
double R_earth = 6.378137e6;

/**
 * @brief Computes gravitational acceleration given ECI position
 * 
 * @param r : velocity vector in ECI [UNITS: m]
 * @return gravitaional acceleration [UNITS: m/s^2]
 */
Vector3 gravitational_acceleration(const Vector3 r);

/**
 * @brief Computes gravitational acceleration asssuming a spherical Earth given ECI position
 * 
 * @param r : velocity vector in ECI [UNITS: m]
 * @return gravitaional acceleration [UNITS: m/s^2]
 */
Vector3 spherical_acceleration(const Vector3 r);

/**
 * @brief Computes J2 gravitational acceleration given ECI position
 * 
 * @param r : velocity vector in ECI [UNITS: m]
 * @return gravitaional acceleration [UNITS: m/s^2]
 */
Vector3 J2_perturbation(const Vector3 r);


#endif   // C___gravity_H