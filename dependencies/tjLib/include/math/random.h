#ifndef _TJLIB_RANDOM_
#define _TJLIB_RANDOM_

#include <random>

#include "EigenWrapper.h"

double randomUniformLog(double min, double max, std::default_random_engine& rng);

template <typename T>
T randomUniform(T min, T max, std::default_random_engine& rng) {
    // can't just do uniform(min, max) since that will fail a static_sssert in
    // <random> when T is a non-floating point type
    std::uniform_real_distribution<double> uniform(0, 1);
    return min + (max - min) * uniform(rng);
}

double randomNormal(double mean, double std_dev, std::default_random_engine& rng);

Vector3 randomUnitVector(std::default_random_engine& rng);

Vector3 randomUnitVectorPerpendicularTo(const Vector3& a, std::default_random_engine& rng);

/**
 * @brief Computes a quaternion from a given rotation angle about a random axis
 *
 * @param angle rotation angle [rad]
 * @param rng the random number generator to use
 * @return Quaternion
 */
Quaternion randomQuaternion(double angle, std::default_random_engine& rng);

#endif   // _TJLIB_RANDOM_