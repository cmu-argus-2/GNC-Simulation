/**
 * @file random.h
 * @author Tushaar Jain (tushaarj@andrew.cmu.edu)
 * @brief Utilitites for generating random values
 * @date 2024-10-09
 *
 */
#ifndef _TJLIB_RANDOM_
#define _TJLIB_RANDOM_

#include <random>

#include "EigenWrapper.h"

/**
 * @brief Sample a random value at uniform after a logarithmic transformation
 *
 * @param min
 * @param max
 * @param rng random number generator
 * @return double 2^x where x~Uniform(log_2(min), log_2(max))
 */
double randomUniformLog(double min, double max, std::default_random_engine& rng);

/**
 * @brief Sample a random value at uniform
 *
 * @tparam T Numeric type to sample
 * @param min
 * @param max
 * @param rng random number generator
 * @return T the random value
 */
template <typename T>
T randomUniform(T min, T max, std::default_random_engine& rng) {
    // can't just do uniform(min, max) since that will fail a static_sssert in
    // <random> when T is a non-floating point type
    std::uniform_real_distribution<double> uniform(0, 1);
    return min + (max - min) * uniform(rng);
}

double randomNormal(double mean, double std_dev, std::default_random_engine& rng);

/**
 * @brief Generates a 3d unit vector pointing in a random direction. Technically not uniform in the directions because
 * we first compute a rando mvector whose entries are drawn from U[-1, 1], so we are effectively sampling over the
 * surface of a cube and then normalizing. The purer, correct way would be to sapmle over the surface of a 3d sphere but
 * that's not what I'm doing.
 *
 * @param rng random number generator
 * @return Vector3 3d unit vector
 */
Vector3 randomUnitVector(std::default_random_engine& rng);

/**
 * @brief Pick a random vector among all unit vectors that are perpendicular to a
 *
 * @param a vector to which we wish to generate a perpendicular unit vector
 * @param rng random nubmer generator
 * @return Vector3 the random unit vector that is perpendicular to a
 */
Vector3 randomUnitVectorPerpendicularTo(const Vector3& a, std::default_random_engine& rng);

/**
 * @brief Computes a quaternion from a given rotation angle about a random axis
 *
 * @param angle rotation angle [rad]
 * @param rng random number generator
 * @return Quaternion
 */
Quaternion randomQuaternion(double angle, std::default_random_engine& rng);

#endif   // _TJLIB_RANDOM_