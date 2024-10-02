#include "RigidBody.h"

#include "math/EigenWrapper.h"

RigidBody::RigidBody(double mass, const Matrix_3x3& InertiaTensor, const Vector3& init_pos_b_wrt_g_in_g,
                     const Quaternion& init_g_q_b, const Vector3& init_vel_b_wrt_g_in_b,
                     const Vector3& init_omega_b_wrt_g_in_b)
    : mass_(mass), InertiaTensor_(InertiaTensor) {
    InertiaTensorInverse_ = InertiaTensor.inverse();

    set_pos_b_wrt_g_in_g(init_pos_b_wrt_g_in_g);
    set_g_q_b(init_g_q_b);
    set_vel_b_wrt_g_in_b(init_vel_b_wrt_g_in_b);
    set_omega_b_wrt_g_in_b(init_omega_b_wrt_g_in_b);

    u_ = Vector6::Zero();
}

Vector3 RigidBody::get_gravity_b() {
    // TODO(Amaar): gravity and J2 computations here
    return Vector3::Zero();
}

void RigidBody::applyForce(const Vector3& force_b, const Vector3& pointOfApplication_b) {
    set_net_force_b(get_net_force_b() + force_b);

    // accounting for the applied body force's resulting moment
    applyMoment(pointOfApplication_b.cross(force_b));
}

void RigidBody::applyMoment(const Vector3& moment_b) {
    set_net_moment_b(get_net_moment_b() + moment_b);
}

void RigidBody::clearAppliedForcesAndMoments() {
    set_net_force_b(Vector3::Zero());
    set_net_moment_b(Vector3::Zero());
}

StateVector f(const StateVector& x, const Vector6& u, const Matrix_3x3& InertiaTensor,
              const Matrix_3x3& InertiaTensorInverse, double mass) {
    StateVector xdot;   // derivative of the state vector

    // aliases for brevity

    auto v     = ::get_vel_b_wrt_g_in_b(x);
    auto q     = ::get_g_q_b(x);
    auto f_b   = ::get_net_force_b(u);
    auto J     = InertiaTensor;
    auto Jinv  = InertiaTensorInverse;
    auto tau_b = ::get_net_moment_b(u);
    auto w     = ::get_omega_b_wrt_g_in_b(x);

    Matrix_4x4 L = Matrix_4x4::Zero();
    L << q.w(), -q.x(), -q.y(), -q.z(),   //
        q.x(), q.w(), -q.z(), q.y(),      //
        q.y(), q.z(), q.w(), -q.x(),      //
        q.z(), -q.y(), q.x(), q.w();

    Matrix_4x3 H = Matrix_4x3::Zero();
    H(1, 0)      = 1;
    H(2, 1)      = 1;
    H(3, 2)      = 1;

    auto G = L * H;

    Vector3 rdot        = q * v;
    Vector4 qdot_coeffs = 0.5 * G * w;
    Quaternion qdot{qdot_coeffs(0), qdot_coeffs(1), qdot_coeffs(2), qdot_coeffs(3)};
    Vector3 vdot = f_b / mass - w.cross(v);
    Vector3 wdot = Jinv * (tau_b - w.cross(J * w));

    // the functions are called set_X but using them to set derivatives of X quantities in the correct positions of the
    // state vector
    ::set_pos_b_wrt_g_in_g(xdot, rdot);
    ::set_g_q_b(xdot, qdot, false);
    ::set_vel_b_wrt_g_in_b(xdot, vdot);
    ::set_omega_b_wrt_g_in_b(xdot, wdot);

    return xdot;
}

StateVector rk4(const StateVector& x, const Vector6& u, const Matrix_3x3& InertiaTensor,
                const Matrix_3x3& InertiaTensorInverse, double mass, double dt) {
    double half_dt    = dt * 0.5;
    StateVector k1    = f(x, u, InertiaTensor, InertiaTensorInverse, mass);
    StateVector k2    = f(x + half_dt * k1, u, InertiaTensor, InertiaTensorInverse, mass);
    StateVector k3    = f(x + half_dt * k2, u, InertiaTensor, InertiaTensorInverse, mass);
    StateVector k4    = f(x + dt * k3, u, InertiaTensor, InertiaTensorInverse, mass);
    StateVector x_new = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // renormalize the attitude quaternion (taken care of by set_g_q_b())
    set_g_q_b(x_new, get_g_q_b(x_new), true);
    return x_new;
}

void RigidBody::set_pos_b_wrt_g_in_g(const Vector3& new_pos_b_wrt_g_in_g) {
    ::set_pos_b_wrt_g_in_g(x_, new_pos_b_wrt_g_in_g);
}
void RigidBody::set_g_q_b(const Quaternion& new_g_q_b) {
    ::set_g_q_b(x_, new_g_q_b, true);
}
void RigidBody::set_b_q_g(const Quaternion& new_b_q_g) {
    ::set_b_q_g(x_, new_b_q_g, true);
}
void RigidBody::set_vel_b_wrt_g_in_b(const Vector3& new_vel_b_wrt_g_in_b) {
    ::set_vel_b_wrt_g_in_b(x_, new_vel_b_wrt_g_in_b);
}
void RigidBody::set_omega_b_wrt_g_in_b(const Vector3& new_omega_b_wrt_g_in_b) {
    ::set_omega_b_wrt_g_in_b(x_, new_omega_b_wrt_g_in_b);
}
void RigidBody::set_net_force_b(const Vector3& new_net_force_b) {
    ::set_net_force_b(u_, new_net_force_b);
}
void RigidBody::set_net_moment_b(const Vector3& new_net_moment_b) {
    ::set_net_moment_b(u_, new_net_moment_b);
}

Vector3 RigidBody::get_pos_b_wrt_g_in_g() {
    return ::get_pos_b_wrt_g_in_g(x_);
}
Quaternion RigidBody::get_g_q_b() {
    return ::get_g_q_b(x_);
}
Quaternion RigidBody::get_b_q_g() {
    return ::get_b_q_g(x_);
}
Vector3 RigidBody::get_vel_b_wrt_g_in_b() {
    return ::get_vel_b_wrt_g_in_b(x_);
}
Vector3 RigidBody::get_omega_b_wrt_g_in_b() {
    return ::get_omega_b_wrt_g_in_b(x_);
}
Vector3 RigidBody::get_net_force_b() {
    return ::get_net_force_b(u_);
}
Vector3 RigidBody::get_net_moment_b() {
    return ::get_net_moment_b(u_);
}

void set_pos_b_wrt_g_in_g(StateVector& x, const Vector3& new_pos_b_wrt_g_in_g) {
    x.block<3, 1>(0, 0) = new_pos_b_wrt_g_in_g;
}
void set_g_q_b(StateVector& x, const Quaternion& new_g_q_b, bool normalize) {
    auto copy = new_g_q_b;
    if (normalize) {
        copy = copy.normalized();
    }
    x(3) = copy.w();
    x(4) = copy.x();
    x(5) = copy.y();
    x(6) = copy.z();
}
void set_b_q_g(StateVector& x, const Quaternion& new_b_q_g, bool normalize) {
    ::set_g_q_b(x, new_b_q_g.inverse(), normalize);
}
void set_vel_b_wrt_g_in_b(StateVector& x, const Vector3& new_vel_b_wrt_g_in_b) {
    x.block<3, 1>(7, 0) = new_vel_b_wrt_g_in_b;
}
void set_omega_b_wrt_g_in_b(StateVector& x, const Vector3& new_omega_b_wrt_g_in_b) {
    x.block<3, 1>(10, 0) = new_omega_b_wrt_g_in_b;
}
void set_net_force_b(Vector6& u, const Vector3& new_net_force_b) {
    u.block<3, 1>(0, 0) = new_net_force_b;
}
void set_net_moment_b(Vector6& u, const Vector3& new_net_moment_b) {
    u.block<3, 1>(3, 0) = new_net_moment_b;
}

Vector3 get_pos_b_wrt_g_in_g(const StateVector& x) {
    return x.block<3, 1>(0, 0);
}
Quaternion get_g_q_b(const StateVector& x) {
    return Quaternion{x(3), x(4), x(5), x(6)};
}
Quaternion get_b_q_g(const StateVector& x) {
    return get_g_q_b(x).inverse();
}
Vector3 get_vel_b_wrt_g_in_b(const StateVector& x) {
    return x.block<3, 1>(7, 0);
}
Vector3 get_omega_b_wrt_g_in_b(const StateVector& x) {
    return x.block<3, 1>(10, 0);
}
Vector3 get_net_force_b(const Vector6& u) {
    return u.block<3, 1>(0, 0);
}
Vector3 get_net_moment_b(const Vector6& u) {
    return u.block<3, 1>(3, 0);
}