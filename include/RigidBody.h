#ifndef C___RIGIDBODY_H
#define C___RIGIDBODY_H

#include "EigenWrapper.h"

/**
 * @brief Assumes body-frame origin coincides with the center-of-mass
 *
 */
class RigidBody {
   public:
    /**
     * @brief Construct a new Rigid Body object
     *
     * @param mass // [kg]
     * @param InertiaTensor in body frame // [kg * m^2]
     * @param init_pos_b_wrt_g_in_g // [m]
     * @param init_g_q_b initial rotation
     * @param init_vel_b_wrt_g_in_b // [m/s]
     * @param init_omega_b_wrt_g_in_b // [rad/s]
     */
    RigidBody(double mass, const Matrix_3x3& InertiaTensor, const Vector3& init_pos_b_wrt_g_in_g,
              const Quaternion& init_g_q_b, const Vector3& init_vel_b_wrt_g_in_b,
              const Vector3& init_omega_b_wrt_g_in_b);

    /**
     * @brief Compute force due to gravity, expressed in body frame
     *
     * @return Vector3 [N]
     */
    Vector3 get_gravity_b();

    Vector3 get_net_force_b();
    Vector3 get_net_moment_b();
    Vector3 get_pos_b_wrt_g_in_g();
    Quaternion get_g_q_b();
    Quaternion get_b_q_g();
    Vector3 get_vel_b_wrt_g_in_b();
    Vector3 get_omega_b_wrt_g_in_b();

    /**
     * @brief Applies a force
     *
     * @param force_b force to apply [N]
     * @param pointOfApplication_b body position at which to apply the force // [m]
     */
    void applyForce(const Vector3& force_b, const Vector3& pointOfApplication_b);

    /**
     * @brief Applies a moment
     *
     * @param moment_b moment to apply [N*m]
     */
    void applyMoment(const Vector3& moment_b);

    void clearAppliedForcesAndMoments();

    /**
     * @brief Fourth order Runge-Kutta integration
     *
     * @param dt integration time [s]
     */
    void rk4(double dt);

   private:
    /**
     * @brief Comprised of 4 parts:
     * (3x1) pos_b_wrt_g_in_g  : position of body from global frame, expressed in global frame [m]
     * (4x1) g_q_b [w,x,y,z]   : rotation that transforms a vector with body coords. into one with global coords.
     * (3x1) vel_b_wrt_g_in_b  : linear velocity of body with respect to global frame, expressed in body frame [m/s]
     * (3x1) omega_b_wrt_g_in_b: angular velocity of body with respect to global frame, expressed in body frame [rad/s]
     */
    StateVector x_;   // state

    /**
     * @brief Net Force & Torque applied:
     * (3x1) net force in body-frame [N]
     * (3x1) net torque in body-frame [N*m]
     */
    Vector6 u_;

    Vector3 gravity_b_;                 // force of gravity in body frame: [N]
    double mass_;                       // mass: [kg]
    Matrix_3x3 InertiaTensor_;          // in body frame [kg*m^2]
    Matrix_3x3 InertiaTensorInverse_;   // in body frame [1/(kg*m^2)]

    void set_net_force_b(const Vector3& new_net_force_b);
    void set_net_moment_b(const Vector3& new_net_moment_b);
    void set_pos_b_wrt_g_in_g(const Vector3& new_pos_b_wrt_g_in_g);
    void set_g_q_b(const Quaternion& new_g_q_b);
    void set_b_q_g(const Quaternion& new_b_q_g);
    void set_vel_b_wrt_g_in_b(const Vector3& new_vel_b_wrt_g_in_b);
    void set_omega_b_wrt_g_in_b(const Vector3& new_omega_b_wrt_g_in_b);

    /**
     * @brief State dynamics
     *
     * @param x state
     * @param u applied forces and torques in body frame
     * @return StateVector state derivative
     */
    StateVector f(const StateVector& x, const Vector6& u);
};

Vector3 get_net_force_b(const Vector6& u);
Vector3 get_net_moment_b(const Vector6& u);
Vector3 get_pos_b_wrt_g_in_g(const StateVector& x);
Quaternion get_g_q_b(const StateVector& x);
Quaternion get_b_q_g(const StateVector& x);
Vector3 get_vel_b_wrt_g_in_b(const StateVector& x);
Vector3 get_omega_b_wrt_g_in_b(const StateVector& x);

void set_net_force_b(Vector6& u, const Vector3& new_net_force_b);
void set_net_moment_b(Vector6& u, const Vector3& new_net_moment_b);
void set_pos_b_wrt_g_in_g(StateVector& x, const Vector3& new_pos_b_wrt_g_in_g);
void set_g_q_b(StateVector& x, const Quaternion& new_g_q_b, bool normalize);
void set_b_q_g(StateVector& x, const Quaternion& new_b_q_g, bool normalize);
void set_vel_b_wrt_g_in_b(StateVector& x, const Vector3& new_vel_b_wrt_g_in_b);
void set_omega_b_wrt_g_in_b(StateVector& x, const Vector3& new_omega_b_wrt_g_in_b);

#endif   // C___RIGIDBODY_H
