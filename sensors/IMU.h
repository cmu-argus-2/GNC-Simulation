#ifndef _IMU_
#define _IMU_

#include <array>
#include <random>

#include "IMUparams.h"
#include "math/EigenWrapper.h"
#include "Sensor.h"
struct IMUSignal {
    Vector3 gyro;
    Vector3 accel;
};

class IMU {
   public:
    IMU(double dt, IMUNoiseParams params, std::default_random_engine& rng);
    IMUSignal update(const IMUSignal& clean);
    IMUSignal getBias();

   private:
    TriAxisSensor gyro_;
    TriAxisSensor accel_;
};

#endif