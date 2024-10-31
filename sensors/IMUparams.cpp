#include "IMUparams.h"

#include <random>

#include "math/random.h"

void printTriParams(TriAxisParams params) {
    std::string axes = "xyz";
    for (int i = 0; i < 3; i++) {
        std::cout << "Axis " << axes[i] << ":" << std::endl;
        params[i].printParams();
    }
}

IMUNoiseParams randomIMUParams(IMUNoiseParams minIMUparams,
                               IMUNoiseParams maxIMUparams,
                               std::default_random_engine& rng) {
    IMUNoiseParams sample{};

    for (int i = 0; i < 3; i++) {
        sample.accel[i].sigma_v = randomUniformLog(
            minIMUparams.accel[i].sigma_v, maxIMUparams.accel[i].sigma_v, rng);
        sample.gyro[i].sigma_v = randomUniformLog(
            minIMUparams.gyro[i].sigma_v, maxIMUparams.gyro[i].sigma_v, rng);

        sample.accel[i].biasParams.bias =
            randomUniform(minIMUparams.accel[i].biasParams.bias,
                          maxIMUparams.accel[i].biasParams.bias, rng);
        sample.gyro[i].biasParams.bias =
            randomUniform(minIMUparams.gyro[i].biasParams.bias,
                          maxIMUparams.gyro[i].biasParams.bias, rng);

        sample.accel[i].biasParams.sigma_w =
            randomUniformLog(minIMUparams.accel[i].biasParams.sigma_w,
                             maxIMUparams.accel[i].biasParams.sigma_w, rng);
        sample.gyro[i].biasParams.sigma_w =
            randomUniformLog(minIMUparams.gyro[i].biasParams.sigma_w,
                             maxIMUparams.gyro[i].biasParams.sigma_w, rng);

        sample.accel[i].biasParams.correlationTime = randomUniformLog(
            minIMUparams.accel[i].biasParams.correlationTime,
            maxIMUparams.accel[i].biasParams.correlationTime, rng);
        sample.gyro[i].biasParams.correlationTime = randomUniformLog(
            minIMUparams.gyro[i].biasParams.correlationTime,
            maxIMUparams.gyro[i].biasParams.correlationTime, rng);

        sample.accel[i].scale_factor_error =
            randomUniform(minIMUparams.accel[i].scale_factor_error,
                          maxIMUparams.accel[i].scale_factor_error, rng);
        sample.gyro[i].scale_factor_error =
            randomUniform(minIMUparams.gyro[i].scale_factor_error,
                          maxIMUparams.gyro[i].scale_factor_error, rng);
    }

    return sample;
}
