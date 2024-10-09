
#include "math/conversions.h"

#include <cmath>

#include "math/EigenWrapper.h"

double wrap_angle(double radians) {
    return radians - (2 * M_PI) * (floor)(radians / (2 * M_PI));
}

double wrap_angle_to_plus_minus_pi(double radians) {
    return wrap_angle(radians) - M_PI;
}

double angle(const Quaternion& q) {
    Quaternion q_normalized = q.normalized();
    if (1.0 < std::fabs(q_normalized.w())) {
        return 0;
    }
    return 2 * std::acos(q_normalized.w());
}

Vector3 axis(const Quaternion& q) {
    return Vector3{q.x(), q.y(), q.z()}.normalized();
}