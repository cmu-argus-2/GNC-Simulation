#include "Bias.h"

#include <random>

#include "IMUparams.h"

Bias::Bias(double dt, BiasParams params, std::default_random_engine& rng)
    : dt_{dt},
      params_{params},
      random_walk_{
          std::normal_distribution<double>{0, params.sigma_w / sqrt(dt)}},
      rng_{rng} {
}

double Bias::update() {
    params_.bias +=
        dt_ * (-params_.bias / params_.correlationTime + random_walk_(rng_));
    return params_.bias;
}

double Bias::getBias() const {
    return params_.bias;
}