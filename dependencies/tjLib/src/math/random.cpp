#include "math/random.h"

#include <random>

#include "math/EigenWrapper.h"

double randomUniformLog(double min, double max, std::default_random_engine& rng) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (min == max) {   // purposely comparing floats
#pragma GCC diagnostic pop
        return min;
    }

    double minlog2 = std::log2(min);
    double maxlog2 = std::log2(max);
    std::uniform_real_distribution<> powerDistribution(minlog2, maxlog2);
    double power = powerDistribution(rng);
    return std::pow(2, power);
}

double randomNormal(double mean, double std_dev, std::default_random_engine& rng) {
    std::normal_distribution<> normal(mean, std_dev);
    return normal(rng);
}

Vector3 randomUnitVector(std::default_random_engine& rng) {
    return Vector3{
        randomUniform(-1.0, 1.0, rng),
        randomUniform(-1.0, 1.0, rng),
        randomUniform(-1.0, 1.0, rng),
    }
        .normalized();
}

Vector3 randomUnitVectorPerpendicularTo(const Vector3& a, std::default_random_engine& rng) {
    auto b = randomUnitVector(rng);
    while (b.cross(a) == Vector3::Zero()) {
        b = randomUnitVector(rng);
    }
    auto c = a.cross(b);
    return c.normalized();
}

Quaternion randomQuaternion(double angle, std::default_random_engine& rng) {
    Vector3 axis = randomUnitVector(rng);
    Quaternion q{cos(angle / 2.0), sin(angle / 2.0) * axis.x(), sin(angle / 2.0) * axis.y(),
                 sin(angle / 2.0) * axis.z()};
    return q;
}
