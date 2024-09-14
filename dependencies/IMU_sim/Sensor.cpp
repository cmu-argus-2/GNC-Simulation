#include "Sensor.h"

#include <random>

#include "EigenWrapper.h"
#include "IMUparams.h"
// #include "rng.h"

Sensor::Sensor(double dt, SensorNoiseParams params,
               std::default_random_engine& rng)
    : bias_{dt, params.biasParams, rng},
      scale_factor_error_{params.scale_factor_error},
      white_noise_{
          std::normal_distribution<double>{0, params.sigma_v / sqrt(dt)}},
      rng_{rng} {
}

double Sensor::update(double clean) {
    bias_.update();
    return (1 + scale_factor_error_) * clean + bias_.getBias() +
           white_noise_(rng_);
}

double Sensor::getBias() {
    return bias_.getBias();
}

// =======================================================================

TriAxisSensor::TriAxisSensor(double dt, TriAxisParams axes_params,
                             std::default_random_engine& rng)
    : x_{dt, axes_params[0], rng},   //
      y_{dt, axes_params[1], rng},   //
      z_{dt, axes_params[2], rng} {
}

Vector3 TriAxisSensor::getBias() {
    return Vector3{x_.getBias(), y_.getBias(), z_.getBias()};
}
Vector3 TriAxisSensor::update(Vector3 clean) {
    return Vector3{x_.update(clean.x()),   //
                   y_.update(clean.y()),   //
                   z_.update(clean.z())};
}
