#ifndef _TJLIB_VECTOR_MATH_
#define _TJLIB_VECTOR_MATH_

#include <cmath>
#include <vector>

#include "EigenWrapper.h"

typedef std::vector<double> vectord;

vectord zeros(unsigned long size);
vectord operator*(double scale, const vectord& x);
vectord operator*(const vectord& x, double scale);
vectord operator-(const vectord& lhs, const vectord& rhs);
vectord operator-(const vectord& x, double offset);
vectord operator+(const vectord& x, double offset);
vectord operator+(const vectord& lhs, const vectord& rhs);
vectord operator*(const vectord& lhs, const vectord& rhs);
vectord sqrt(const vectord& x);
vectord diff(const vectord& x);
vectord abs(const vectord& x);
double mean(const vectord& x);

/**
 * @brief Computes scalar Cov(x,y) (no Bessel correction)
 *
 * @param x vector of values
 * @param y vector of values
 * @return double Cov(x)
 */
double covariance(const vectord& x, const vectord& y);

/**
 * @brief Computes the variance of population x (no Bessel correction)
 *
 * @param x vector of values
 * @return double Var(x)
 */
double variance(const vectord& x);

/**
 * @brief Computes the covariance matrix of x and y (no Bessel correction)
 *
 * @param x vector of vectors
 * @return Eigen::MatrixXd Cov(x)
 */
Eigen::MatrixXd covarianceMatrix(const std::vector<vectord>& x);

// INCLUDES END!
// assumes 1 < N
vectord linspace(double start, double end, size_t N);

// INCLUDES END if possible
vectord linspace_delta(double start, double end, double delta);

// Function to perform linear interpolation
template <typename T>
T interpolate(double x, double x0, double x1, T y0, T y1) {
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

/**
 * @brief Resample a data series according to a new sampling scheme. Uses
 * interpolation
 *
 * @param [in] originalTimes original time series
 * @param [in] originalData original data series corresponding to original time
 * series
 * @param [in] initialSampleTime the new initial sampling time
 * @param [in] sampleDT the new sampling period
 *
 * @param [out] newTimes new, resampled time series
 * @param [out] newData new, resampled data series corresponding to new time
 * series
 *
 * Assumes originalTimes and newTimes don't alias in memory
 * Assumes originalData and newData don't alias in memory
 */
template <typename DataType, typename DataSeriesType>
void resampleTimeSeries(const vectord& originalTimes, const DataSeriesType& originalData, const vectord& newTimes,
                        DataSeriesType& newData) {
    size_t j = 0;
    for (double newTime : newTimes) {
        if (newTime < originalTimes[0]) {
            newData.push_back(originalData[0]);   // repeat first element
        } else if (originalTimes[originalTimes.size() - 1] < newTime) {
            newData.push_back(originalData[originalTimes.size() - 1]);   // repeat last element
        } else {
            // Find the two original time points surrounding the new time
            while (j < originalTimes.size() - 1 && originalTimes[j + 1] < newTime) {
                j++;
            }

            // Perform linear interpolation between the two surrounding data
            // points
            if (j < originalTimes.size() - 1) {
                double x0   = originalTimes[j];
                double x1   = originalTimes[j + 1];
                DataType y0 = originalData[j];
                DataType y1 = originalData[j + 1];

                DataType newSample = interpolate<DataType>(newTime, x0, x1, y0, y1);
                newData.push_back(newSample);
            }
        }
    }
}

Quaternion operator*(double scale, const Quaternion& q);
Quaternion operator*(const Quaternion& q, double scale);
Quaternion operator/(const Quaternion& q, double scale);
Quaternion operator+(const Quaternion& lhs, const Quaternion& rhs);
Quaternion operator-(const Quaternion& lhs, const Quaternion& rhs);
std::vector<Quaternion> operator/(const std::vector<Quaternion>& lhs, const std::vector<Quaternion>& rhs);
void normalize(std::vector<Quaternion> orientations);

#endif