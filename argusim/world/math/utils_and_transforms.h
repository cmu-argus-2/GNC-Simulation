
#ifndef _POSE_EKF_UTILS_AND_TRANSFORMS_
#define _POSE_EKF_UTILS_AND_TRANSFORMS_

#include "math/EigenWrapper.h"
#include "SpiceUsr.h"
#include <random> 

// How close each A(i,j) must be to A(j,i) for matrix to be considered symmetric
static constexpr double VALID_SYMMETRIC_MATRIX_TOLERANCE = 1E-60;

// Divide by zero tolerance
static constexpr double DIVIDE_BY_ZERO_TOLERANCE = 1E-5;

#define DEG_2_RAD(x) ((x) * (M_PI / 180.0))
#define RAD_2_DEG(x) ((x) * (180.0 / M_PI))

/**
 * @brief Converts seconds since the unix epoch to seconds since J2000 epoch
 *
 * @param unixSeconds # of seconds past the unix epoch (Jan 1, 1970 0:0:0)
 * @return int64_t # of seconds past the J2000 epoch (Jan 1, 2000 11:58:55.816
 * AM)
 */
int64_t unixToJ2000(int64_t unixSeconds);

void set_b_R_n(StateVector &state, const Quaternion &new_b_R_n);
void set_n_R_b(StateVector &state, const Quaternion &new_n_R_b);
void setGyroBias(StateVector &state, const Vector3 &newGyroBias);
void setAccelBias(StateVector &state, const Vector3 &newAccelBias);

Quaternion get_b_R_n(const StateVector &state);
Quaternion get_n_R_b(const StateVector &state);
Vector3 getGyroBias(const StateVector &state);
Vector3 getAccelBias(const StateVector &state);

/**
 * @brief Compute the skew-symmetric, 3x3 matrix corresponding to
 * cross product with v
 *
 * @param v vector on the left side of the hypothetical cross product
 * @return Matrix_3x3 3x3 skew symmetric matrix
 */
Matrix_3x3 toSkew(const Vector3 &v);

/**
 * @brief Zeros-out really small values of the rotation matrix because they
 * probably should've been 0 to begin with
 *
 * @param R rotation matrix to be cleaned up
 * @return Matrix_3x3 cleaned up rotation matrix
 */
Matrix_3x3 cleanRotMatrix(Matrix_3x3 R);

/**
 * @brief Defines a random rotation about a random axis
 * 
 * @param dist : Normal Distribution characterizing the noise profile
 * @param gen : seeded random generator
 * @return 3x3 matrix represnting a random rotation
 */
Matrix_3x3 random_SO3_rotation(std::normal_distribution<double> dist, std::mt19937 gen);

/**
 * @brief Get the ENU to ECEF transform based on latitude and longitude of the
 * landing site
 *
 * @param latitude_deg latitude of landing site [degrees]
 * @param longitude_deg longitude of landing site [degrees]
 * @return Matrix_3x3 the ECEF_R_ENU transform
 */
Matrix_3x3 get_ECEF_R_ENU(double latitude_deg, double longitude_deg);

/**
 * @brief Convert quaternion to constituent euler angle sequence given by the
 * decomposition: q = R_x(roll) * R_y(pitch) * R_z(yaw)
 * https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf#page=7
 * yaw is in [-pi, pi]
 * pitch is in [-pi/2, pi/2]
 * roll is in [-pi, pi]
 *
 * @param q rotation
 * @return Vector3 {yaw, pitch, roll} [radians]
 */
Vector3 intrinsic_xyz_decomposition(const Quaternion &q);

/**
 * @brief Convert quaternion to constituent euler angle sequence given by the
 * decomposition: q = R_z(yaw) * R_y(pitch) * R_x(roll)
 * https://web.mit.edu/2.05/www/Handout/HO2.PDF
 * yaw is in [-pi, pi]
 * pitch is in [-pi/2, pi/2]
 * roll is in [-pi, pi]
 *
 * @param q rotation
 * @return Vector3 {yaw, pitch, roll} [radians]
 */
Vector3 intrinsic_zyx_decomposition(const Quaternion &q);


// CSPICE COORDINATE TRANSFORMS

/**
 * @brief Load all kernels from datapaths necessary for SPICE
 *
 * TOREAD : types of kernel according to SPICE [read https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Kernel%20Types]
 */
void loadAllKernels();

/**
 * @brief Cast a double[3][3] into an Eigen <double, 3, 3> matrix
 *
 * @param M SpiceDouble 3x3 matrix
 * @return R Eigen 3x3 matrix
 */
Matrix_3x3 Cspice2Eigen(SpiceDouble M[3][3]);

/**
 * @brief Computes the rotation matrix from ECI to ECEF at a given time
 *
 * @param t_J2000 - seconds past J2000 i.e., seconds past Jan 1st 2000, 12:00:00 PM
 * @return R Eigen 3x3 matrix representing roation from ECI to ECEF
 */
Matrix_3x3 ECI2ECEF(double t_J2000);

/**
 * @brief Computes the rotation matrix from ECEF to ECI at a given time
 *
 * @param t_J2000 - seconds past J2000 i.e., seconds past Jan 1st 2000, 12:00:00 PM
 * @return R Eigen 3x3 matrix representing roation from ECEF to ECI
 */
Matrix_3x3 ECEF2ECI(double t_J2000);

/**
 * @brief Transforms a vector in ECEF frame to Geodetic coordinates
 *
 * @param v_ecef - vector in ECEF frame [UNITS : m]
 * @return vector in Geodetic coordinates (long, lat, alt) [rad, rad, m]
 */
Vector3 ECEF2GEOD(Vector3 v_ecef);

/**
 * @brief Transforms a vector in SEZ frame to ECEF
 *
 * @param r_sez - vector in ECEF frame
 * @param latitude - latitude where SEZ vector is constructed
 * @param longitude - longitude where SEZ vector is constructed
 * @return vector in ECEF coordinates
 */
Vector3 SEZ2ECEF(Vector3 r_sez, double latitude, double longitude);

/**
 * @brief Transforms a vector in ECI frame to Geodetic(LLA) coordinates
 *
 * @param v_eci - vector in ECI frame
 * @param t_J2000 - Time of conversion as seconds past J2000
 * @return vector in Geodetic coordinates
 */
Vector3 ECI2GEOD(Vector3 v_eci, double t_J2000);

/**
 * @brief Converts Keplerian Elements into ECI state
 *
 * @param KOE - Keplerian Orbital element vector
 * @param t_J2000 - seconds past J2000
 * @return position and velocity vector in ECI coordinates
 */
Vector6 KOE2ECI(Vector6 KOE, double t_J2000);

/**
 * @brief Converts time from seconds past J2000 to UTC date vector
 *
 * @param t_J2000 - seconds past J2000
 * @return UTC date as a 6 element vector [Y, doy, h, m, s]
 */
Vector5 TJ2000toUTC(double t_J2000);

/**
 * @brief Converts time from seconds past J2000 to UTC date string
 *
 * @param t_J2000 - seconds past J2000
 * @return UTC datestring in ISOC format
 */
std::string TJ2000toUTCString(double t_J2000);

/**
 * @brief Converts time from UTC string in format 'YYYY-MM-DD HH:MM:SS' to seconds past J2000
 *
 * @param UTC - UTC string
 * @return t_J2000 : seconds past J2000
 */
double UTCStringtoTJ2000 (std::string UTC);

#endif   //_POSE_EKF_UTILS_AND_TRANSFORMS_ header