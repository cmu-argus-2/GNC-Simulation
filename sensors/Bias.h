#ifndef __BIAS
#define __BIAS

#include <random>

#include "IMUparams.h"

class Bias {
   private:
    double dt_;   // [s]
    BiasParams params_;
    std::normal_distribution<double> random_walk_;
    std::default_random_engine& rng_;

   public:
    /**
     * @brief Construct a new Bias object
     *
     * @param dt sample period [s]
     * @param BiasParams bias params [units]
     * @param rng rng object
     */
    Bias(double dt, BiasParams params, std::default_random_engine& rng);
    double update();
    double getBias() const;
};

#endif