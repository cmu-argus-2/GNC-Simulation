#include "IMU.h"

#include <random>

#include "tjLib/math/EigenWrapper.h"
#include "IMUparams.h"

IMU::IMU(double dt, IMUNoiseParams params, std::default_random_engine& rng)
    : gyro_{dt, params.gyro, rng}, accel_{dt, params.accel, rng} {
}

IMUsignal IMU::getBias() {
    IMUsignal bias;
    bias.gyro  = gyro_.getBias();
    bias.accel = accel_.getBias();
    return bias;
}

IMUsignal IMU::update(const IMUsignal& clean) {
    IMUsignal measurement;
    measurement.gyro  = gyro_.update(clean.gyro);
    measurement.accel = accel_.update(clean.accel);
    return measurement;
}
