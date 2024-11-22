#include "math/vector_math.h"

#include <cassert>
#include <vector>

#include "math/EigenWrapper.h"

vectord zeros(unsigned long size) {
    vectord z;
    z.reserve(size);

    for (unsigned int i = 0; i < size; i++) {
        z.push_back(0.0);
    }
    return z;
}

vectord operator*(double scale, const vectord &x) {
    vectord y;
    y.reserve(x.size());

    for (double val : x) {
        y.push_back(scale * val);
    }
    return y;
}

vectord operator*(const vectord &x, double scale) {
    return scale * x;
}

vectord operator-(const vectord &lhs, const vectord &rhs) {
    assert(lhs.size() == rhs.size());
    size_t N = lhs.size();

    vectord y;
    y.reserve(lhs.size());

    for (size_t i = 0; i < N; i++) {
        y.push_back(lhs[i] - rhs[i]);
    }
    return y;
}

vectord operator+(const vectord &x, double offset) {
    vectord y;
    y.reserve(x.size());
    for (auto x_value : x) {
        y.push_back(x_value + offset);
    }
    return y;
}

vectord operator-(const vectord &x, double offset) {
    return x + (-offset);
}

vectord operator+(const vectord &lhs, const vectord &rhs) {
    assert(lhs.size() == rhs.size());
    size_t N = lhs.size();

    vectord y;
    y.reserve(lhs.size());

    for (size_t i = 0; i < N; i++) {
        y.push_back(lhs[i] + rhs[i]);
    }
    return y;
}

vectord operator*(const vectord &lhs, const vectord &rhs) {
    assert(lhs.size() == rhs.size());
    size_t N = lhs.size();

    vectord y;
    y.reserve(lhs.size());

    for (size_t i = 0; i < N; i++) {
        y.push_back(lhs[i] * rhs[i]);
    }
    return y;
}

vectord sqrt(const vectord &x) {
    vectord y;
    y.reserve(x.size());

    for (double val : x) {
        y.push_back(sqrt(val));
    }
    return y;
}

vectord diff(const vectord &x) {
    vectord y;
    y.reserve(x.size() - 1);

    uint N = x.size() - 1;
    for (uint i = 0; i < N; i++) {
        y.push_back(x[i + 1] - x[i]);
    }
    return y;
}

vectord abs(const vectord &x) {
    vectord y;
    y.reserve(x.size() - 1);

    uint N = x.size() - 1;
    for (uint i = 0; i < N; i++) {
        y.push_back(fabs(x[i]));
    }
    return y;
}

double mean(const vectord &x) {
    double total = 0;
    for (double value : x) {
        total += value;
    }
    return total / x.size();
}

double covariance(const vectord &x, const vectord &y) {
    return mean((x - mean(x)) * (y - mean(y)));
}

double variance(const vectord &x) {
    return covariance(x, x);
}

Eigen::MatrixXd covarianceMatrix(const std::vector<vectord> &x) {
    int N = x.size();

    Eigen::MatrixXd P(N, N);

    for (int row = 0; row < N; row++) {
        for (int col = row; col < N; col++) {
            double cov  = covariance(x[row], x[col]);
            P(row, col) = cov;
            if (col != row) {
                P(col, row) = cov;
            }
        }
    }
    return P;
}

vectord linspace(double start, double end, size_t N) {
    assert(1 < N);

    vectord result;
    for (size_t i = 0; i < N; i++) {
        result.push_back(start * (double)(N - 1 - i) / (double)(N - 1) + end * (double)(i) / (double)(N - 1));
    }
    return result;
}

vectord linspace_delta(double start, double end, double delta) {
    bool increasing = (start <= end) and 0 < delta;
    bool decreasing = (start >= end) and 0 > delta;
    assert(increasing or decreasing);

    vectord result;
    double value = start;
    while ((increasing and value <= end) or (decreasing and value >= end)) {
        result.push_back(value);
        value += delta;
    }

    double tolerance = abs(1e-10 * delta);
    if (abs(value - end) < tolerance) {
        result.push_back(end);
    }
    return result;
}

Quaternion operator*(double scale, const Quaternion &q) {
    return Quaternion{scale * q.w(), scale * q.x(), scale * q.y(), scale * q.z()};
}
Quaternion operator*(const Quaternion &q, double scale) {
    return scale * q;
}

Quaternion operator/(const Quaternion &q, double scale) {
    return Quaternion{q.w() / scale, q.x() / scale, q.y() / scale, q.z() / scale};
}

Quaternion operator+(const Quaternion &lhs, const Quaternion &rhs) {
    return Quaternion{lhs.w() + rhs.w(), lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z()};
}

Quaternion operator-(const Quaternion &lhs, const Quaternion &rhs) {
    return Quaternion{lhs.w() - rhs.w(), lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z()};
}

std::vector<Quaternion> operator/(const std::vector<Quaternion> &lhs, const std::vector<Quaternion> &rhs) {
    assert(lhs.size() == rhs.size());
    size_t N = lhs.size();

    std::vector<Quaternion> y;
    y.reserve(lhs.size());

    for (size_t i = 0; i < N; i++) {
        y.push_back(lhs[i] * rhs[i].inverse());
    }
    return y;
}

void normalize(std::vector<Quaternion> orientations) {
    size_t N = orientations.size();
    for (size_t i = 0; i < N; i++) {
        orientations[i].normalize();
    }
}