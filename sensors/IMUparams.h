#ifndef _IMU_PARAMS_
#define _IMU_PARAMS_
#include <array>
#include <iostream>
#include <random>

// Parameters for a first-Order Gauss-Markov model of a bias
struct BiasParams {
    double bias;              // [units]
    double sigma_w;           // additive white noise to time-derivative of bias
                              // [(units/s)/sqrt(Hz)]
    double correlationTime;   // time correlation [s]

    void printParams() const {
        std::cout << "bias: " << bias << std::endl;
        std::cout << "sigma_w: " << sigma_w << std::endl;
        std::cout << "correlationTime: " << correlationTime << std::endl;
    }
};

struct SensorNoiseParams {
    BiasParams biasParams;

    // additive white noise to sensor output [units/sqrt(Hz)]
    double sigma_v{0};

    // multiplier [-]
    double scale_factor_error{0};

    void printParams() const {
        biasParams.printParams();
        std::cout << "sigma_v: " << sigma_v << std::endl;
        std::cout << "scale factor error: " << scale_factor_error << std::endl;
    }
};

typedef std::array<SensorNoiseParams, 3> TriAxisParams;

void printTriParams(TriAxisParams params);
struct IMUNoiseParams {
    TriAxisParams gyro;
    TriAxisParams accel;

    void printParams() const {
        std::cout << "GYRO:" << std::endl;
        printTriParams(gyro);

        std::cout << "ACCELEROMETER:" << std::endl;
        printTriParams(accel);
        std::cout << std::endl;
    }
};

IMUNoiseParams randomIMUParams(IMUNoiseParams minIMUparams,
                               IMUNoiseParams maxIMUparams,
                               std::default_random_engine& rng);
#endif