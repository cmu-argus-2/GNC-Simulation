#include "RigidBody.h"
#include <iostream>

#include "math/EigenWrapper.h"
#include "math/vector_math.h"
#include "ParameterParser.h"
#include "Magnetorquer.h"
#include "utils_and_transforms.h"

#include "gravity.h"
#include "drag.h"
#include "SRP.h"
#include "MagneticField.h"


#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif

VectorXd f(const VectorXd& x, const VectorXd& u, Simulation_Parameters sc, double t_J2000) 
{

    auto xdot = OrbitalDynamics(x, sc.mass, sc.Cd, sc.CR, sc.A, sc.useDrag, sc.useSRP, t_J2000);
    xdot = xdot + AttitudeDynamics(x, u, sc.num_MTBs, sc.num_RWs, sc.G_rw_b, sc.G_mtb_b, sc.I_rw, sc.I_sat, sc.MTB, t_J2000);

    return xdot;
}

VectorXd OrbitalDynamics(const VectorXd& x, double mass, double Cd, double CR, double A, 
                                bool useDrag, bool useSRP, double t_J2000)
{
    VectorXd xdot = VectorXd::Zero(x.size());

    // Extract elements from state vector
    Vector3 r = x(Eigen::seqN(0, 3));
    Vector3 v = x(Eigen::seqN(3, 3));
    Quaternion q{x(6), x(7), x(8), x(9)};

    // Physics Models
    Vector3 vdot = gravitational_acceleration(r);

    
    if (useDrag){ 
        vdot = vdot + drag_acceleration(r, v, q, t_J2000, Cd, A, mass);
    }

    if (useSRP){
        vdot = vdot + SRP_acceleration(r ,q, t_J2000, CR, A, mass);
    }
    
    // Pack acceleration back into the state vector
    xdot(Eigen::seqN(0,3)) = v;
    xdot(Eigen::seqN(3,3)) = vdot;

    return xdot;
}

VectorXd AttitudeDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             const Eigen::MatrixXd& G_rw_b, const Eigen::MatrixXd& G_mtb_b,
                             double I_rw, const Matrix_3x3 I_sat, Magnetorquer MTB, double t_J2000)
{
    
    // Assert matrix sizes
    assert(u.size() == (num_MTBs + num_RWs+1)); // num_MTB + num_RWs torques + Jetson ON/OFF
    assert(G_rw_b.rows() == 3); // Orientation matrix has 3 element vectors
    assert(G_rw_b.cols() == num_RWs); // 1 column for each RW
    assert(G_mtb_b.rows() == 3); // 3D vector for each MTB
    assert(G_mtb_b.cols() == num_MTBs); // 1 column for each MTB

    VectorXd xdot = VectorXd::Zero(x.size());

    // Extract elements of state vector
    Vector3 r = x(Eigen::seqN(0, 3));
    Quaternion q{x(6), x(7), x(8), x(9)}; // initialize attitude quaternion
    q.normalize();
    Vector3 omega{x(10), x(11), x(12)};
    VectorXd omega_rw(num_RWs);
    omega_rw = x(Eigen::seqN(19, num_RWs));
    
    /* Attitude Dynamics */
    Quaternion omega_quat {0, omega(0), omega(1), omega(2)};
    Quaternion qdot_quat = 0.5*q*omega_quat;
    Vector4 qdot{qdot_quat.w(), qdot_quat.x(), qdot_quat.y(), qdot_quat.z()};

    // Reaction Wheels
    auto h_rw = I_rw*omega_rw;
    auto tau_rw = u(Eigen::seqN(num_MTBs, num_RWs));
    
    // Magnetorquers
    Vector3 tau_mtb = MTB.getTorque(u(Eigen::seqN(0,num_MTBs)), q, MagneticField(r, t_J2000));

    // Gyrostat Equation
    Vector3 h_sc = I_sat*omega + G_rw_b*h_rw;
    Vector3 omega_dot = I_sat.inverse()*(-omega.cross(h_sc) - G_rw_b*tau_rw + tau_mtb);
    
    // Reaction wheel speeds
    auto omega_dot_rw = tau_rw/I_rw;
    
    // Pack into state derivative vector
    xdot(Eigen::seqN(6,4)) = qdot;
    xdot(Eigen::seqN(10,3)) = omega_dot;
    xdot(Eigen::seqN(19,num_RWs)) = omega_dot_rw;

    return xdot;
}

VectorXd rk4(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC, double t_J2000, double dt) 
{
    VectorXd x_new(x.size());
    double half_dt    = dt * 0.5;

    auto k1    = f(x, u, SC, t_J2000);
    auto k2    = f(x + half_dt * k1, u, SC, t_J2000 + half_dt);
    auto k3    = f(x + half_dt * k2, u, SC, t_J2000 + half_dt);
    auto k4    = f(x + dt * k3, u, SC, t_J2000 + dt);
    x_new = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // renormalize the attitude quaternion
    x_new(Eigen::seqN(6, 4)) = x_new(Eigen::seqN(6, 4))/x_new(Eigen::seqN(6, 4)).norm();
    // sun position
    x_new(Eigen::seqN(13, 3)) = sun_position_eci(t_J2000 + dt);
    // magnetic field 
    x_new(Eigen::seqN(16, 3)) = MagneticField( x_new(Eigen::seqN(0, 3)), t_J2000 + dt);

    return x_new;
}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pyphysics, m) {
    m.doc() = "pybind11 physics plugin";   // module docstring    

    m.def("rk4", &rk4, "rk4 integrator");
    m.def("ECI2GEOD", &ECI2GEOD, "ECI to Geodetic Coordinates");
}
#endif