#include <iostream>
#include <cmath>
#include <Eigen>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>


Eigen::VectorXd IntegrateWithRK4(
    Eigen::VectorXd (*const f)(
        const Eigen::Ref<const Eigen::VectorXd>& x,
        const Eigen::Ref<const Eigen::VectorXd>& u),
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const Eigen::Ref<const Eigen::VectorXd>& u,
    const double dt
) {
    const Eigen::VectorXd k1{ f(x0, u) };
    const Eigen::VectorXd k2{ f(x0 + dt/2 * k1, u) };
    const Eigen::VectorXd k3{ f(x0 + dt/2 * k2, u) };
    const Eigen::VectorXd k4{ f(x0 + dt * k3, u) };
    return x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
}

Eigen::VectorXd IntegrateWithRK4(
    Eigen::VectorXd (*const f)(const Eigen::Ref<const Eigen::VectorXd>& x),
    const Eigen::Ref<const Eigen::VectorXd>& x0,
    const double dt
) {
    const Eigen::VectorXd k1{ f(x0) };
    const Eigen::VectorXd k2{ f(x0 + dt/2 * k1) };
    const Eigen::VectorXd k3{ f(x0 + dt/2 * k2) };
    const Eigen::VectorXd k4{ f(x0 + dt * k3) };
    return x0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
}


Eigen::VectorXd GetVelocityAndAcceleration(
    const Eigen::Ref<const Eigen::VectorXd>& posAndVel
) {
    const double mu{3.9860044188e14};

    const Eigen::Vector3d r{posAndVel.head(3)};
    const Eigen::Vector3d v{posAndVel.tail(3)};
    const Eigen::Vector3d accGrav{ -mu*r / std::pow(r.norm(), 3) };
    const Eigen::Vector3d accDrag{Eigen::Vector3d::Zero(3)};

    Eigen::VectorXd velAndAcc(posAndVel.size());
    velAndAcc << v, accGrav + accDrag;
    return velAndAcc;
}

Eigen::VectorXd StepPositionAndVelocity(
    const Eigen::Ref<const Eigen::VectorXd>& posAndVel,
    const double dt
) {
    return IntegrateWithRK4(&GetVelocityAndAcceleration, posAndVel, dt);
}


Eigen::VectorXd GetAttitudeRateAndAngularAcceleration(
    const Eigen::Ref<const Eigen::VectorXd>& attAndAngVel,
    const Eigen::Ref<const Eigen::VectorXd>& torque
) {
    Eigen::Quaterniond q;
    q.w() = attAndAngVel(0);
    q.vec() = attAndAngVel(Eigen::seq(1,3));

    const Eigen::Vector3d angVel{attAndAngVel.tail(3)};
    Eigen::Quaterniond w_quat;
    w_quat.w() = 0.0;
    w_quat.vec() = 0.5 * angVel;
    const Eigen::Quaterniond attRate{q * w_quat};

    // Moment of inertia tensor
    const Eigen::DiagonalMatrix<double, 3> J(0.006, 0.001, 0.006);
    const Eigen::Vector3d angAcc{
        J.inverse() * (torque - angVel.cross(J * angVel))};
    //std::cout << angAcc << "\n\n\n";
    Eigen::VectorXd attRateAndAngAcc(attAndAngVel.size());
    attRateAndAngAcc << attRate.w(), attRate.vec(), angAcc;
    return attRateAndAngAcc;
}

Eigen::VectorXd StepAttitudeAndAngularVelocity(
    const Eigen::Ref<const Eigen::VectorXd>& attAndAngVel,
    const Eigen::Ref<const Eigen::VectorXd>& torque,
    const double dt
) {
    return IntegrateWithRK4(&GetAttitudeRateAndAngularAcceleration, attAndAngVel, torque, dt);
}


PYBIND11_MODULE(dynamics, m) {
    m.doc() = "Dynamics function bindings";
    m.def("step_position_and_velocity", &StepPositionAndVelocity, "Discrete-time translational dynamics");
    m.def("step_attitude_and_angular_velocity", &StepAttitudeAndAngularVelocity, "Discrete-time angular dynamics");
}
