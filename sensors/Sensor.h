#ifndef _SENSOR_
#define _SENSOR_

#include <random>

#include "Bias.h"
#include "math/EigenWrapper.h"
#include "IMUparams.h"

class Sensor {
   public:
    Sensor(double dt, SensorNoiseParams params,
           std::default_random_engine& rng);
    double update(double clean);
    double getBias();

   private:
    Bias bias_;
    double scale_factor_error_{0};
    SensorNoiseParams noise_params_;
    std::normal_distribution<double> white_noise_;
    std::default_random_engine& rng_;
};
class TriAxisSensor {
   public:
    TriAxisSensor(double dt, TriAxisParams params,
                  std::default_random_engine& rng);
    Vector3 update(Vector3 clean);
    Vector3 getBias();

   private:
    Sensor x_;
    Sensor y_;
    Sensor z_;
};

#endif