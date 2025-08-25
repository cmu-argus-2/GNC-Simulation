#ifndef _TJLIB_CONVERSIONS_
#define _TJLIB_CONVERSIONS_

#include "EigenWrapper.h"

#define DEG_2_RAD(x) ((x) * (M_PI / 180.0))
#define RAD_2_DEG(x) ((x) * (180.0 / M_PI))
static constexpr double HR_PER_SEC = (1 / 3600.0);

/**
 * @brief wrap radians from [-inf, inf] to [0, 2*pi)
 *
 * @param radians unwrapped angle
 * @return double wrapped angle
 */
double wrap_angle(double radians);

/**
 * @brief wrap degrees from [-inf, inf] to [-pi, pi)
 *
 * @param radians unwrapped angle
 * @return double wrapped angle
 */
double wrap_angle_to_plus_minus_pi(double radians);

/**
 * @brief Get the quaternion's rotation angle in the range [0, pi]
 *
 * @param q quaternion. Normalized internally.
 * @return double the angle [radians]
 */
double angle(const Quaternion& q);

/**
 * @brief Get the quaternion's rotation axis as a unit vector
 *
 * @param q quaternion. Normalized internally.
 * @return Vector3 the axis
 */
Vector3 axis(const Quaternion& q);

/**
 * @brief Convert a 4D vector to a quaternion
 *
 * @param q_vec 4D vector representing the quaternion (w, x, y, z)
 * @return Quaternion the corresponding quaternion
 */
Quaternion vectorToQuaternion(const Vector4& q_vec);

// TODO(tushaar): add sgn(x) to return -1 or 1 based on whether x is negative or
// positive
#endif