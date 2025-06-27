#ifndef C___gravity_H
#define C___gravity_H

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"
/*
#include <GeographicLib/GravityModel.hpp>

using namespace std;
using namespace GeographicLib;
*/

/* CONSTANTS */
double mu = 3.986004415e14;
double J2 = 1.08262668e-3;
double R_earth = 6.3781363e6;
double mu_sun = 1.32712440018e20; 
double mu_moon = 4.902801e12;

/**
 * @brief Computes gravitational acceleration given ECI position
 * 
 * @param r : velocity vector in ECI [UNITS: m]
 * @param t_J2000 : time in J2000 format
 * @param Nmax : maximum degree of the spherical harmonic model
 * @param Mmax : maximum order of the spherical harmonic model
 * @return gravitaional acceleration [UNITS: m/s^2]
 */
Vector3 gravitational_acceleration(const Vector3 r, double t_J2000, int Nmax, int Mmax);

/**
 * @brief Computes gravity gradient torque given ECI position and satellite inertia matrix
 * 
 * @param r : position vector in ECI [UNITS: m]
 * @param I_sat : satellite inertia matrix [UNITS: kg*m^2]
 * @return Gravity gradient torque vector [UNITS: Nm]
 */
Vector3 gravity_gradient_torque(const Vector3& r, const Matrix_3x3& I_sat);

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

/**
 * @brief Computes gravitational acceleration due to the Sun given ECI position and Sun position
 * 
 * @param r : velocity vector in ECI [UNITS: m]
 * @param sun_position : Sun position vector in ECI [UNITS: m]
 * @return gravitaional acceleration due to the Sun [UNITS: m/s^2]
 */
Vector3 sun_gravity(const Vector3 r, const Vector3 sun_position);

/**
 * @brief Computes gravitational acceleration due to the Moon given ECI position and time
 * 
 * @param r : velocity vector in ECI [UNITS: m]
 * @param t_J2000 : time in J2000 format
 * @return gravitaional acceleration due to the Moon [UNITS: m/s^2]
 */
Vector3 moon_gravity(const Vector3 r, double t_J2000);

/**
 * @brief Computes the position of the Moon in ECI coordinates at a given time
 * 
 * @param t_J2000 : time in J2000 format
 * @return Moon position vector in ECI [UNITS: m]
 */
Vector3 moon_position_eci(double t_J2000);

#endif   // C___gravity_H