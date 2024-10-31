#include "IMU.h"

#include <random>

#include "math/EigenWrapper.h"
#include "IMUparams.h"

IMU::IMU(double dt, IMUNoiseParams params, std::default_random_engine& rng)
    : gyro_{dt, params.gyro, rng}, accel_{dt, params.accel, rng} {
}

IMUSignal IMU::getBias() {
    IMUSignal bias;
    bias.gyro  = gyro_.getBias();
    bias.accel = accel_.getBias();
    return bias;
}

IMUSignal IMU::update(const IMUSignal& clean) {
    IMUSignal measurement;
    measurement.gyro  = gyro_.update(clean.gyro);
    measurement.accel = accel_.update(clean.accel);
    return measurement;
}
