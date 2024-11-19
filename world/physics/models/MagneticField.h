#include "math/EigenWrapper.h"

extern "C"
{
    extern void igrf13syn_(int*, double*, int*, double*, double*, double*,double*, double*, double*, double*); // Fortran version of IGRF13 model
}

/**
 * @brief Compute magnetic field at satellite location in J2000 ECI frame
 * 
 * @param r : position vector in ECI frame [UNITS: m]
 * @param t_J2000 : seconds since J2000 used to compute sun position
 * @return magnetic field in ECI frame [UNITS: T]
 */
Vector3 MagneticField(const Vector3 r, double t_J2000);

/**
 * @brief Calls fortran code for local magnetic field in South-East-Zenith frame
 * 
 * @param r_geod : position vector in geodetic frame [UNITS: [rad, rad, m]]
 * @param year : fractional year for the current day
 * @return magnetic field in SEZ frame [UNITS: nT]
 */
Vector3 MagneticFieldSEZ(const Vector3 r_geod, double year);