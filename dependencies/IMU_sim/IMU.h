#ifndef _IMU_
#define _IMU_

#include <array>
#include <random>

#include "EigenWrapper.h"
#include "IMUparams.h"
#include "Sensor.h"
struct IMUsignal {
    Vector3 gyro;
    Vector3 accel;
};

class IMU {
   public:
    IMU(double dt, IMUNoiseParams params, std::default_random_engine& rng);
    IMUsignal update(const IMUsignal& clean);
    IMUsignal getBias();

   private:
    TriAxisSensor gyro_;
    TriAxisSensor accel_;
};

#endif