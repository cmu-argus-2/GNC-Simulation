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
#include "power.h"
#include <random>


#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif

VectorXd f(const VectorXd& x, const VectorXd& u, Simulation_Parameters sc, double t_J2000) 
{
     
    auto xdot = OrbitalDynamics(x, sc.mass, sc.Cd, sc.CR, sc.A, sc.useDrag, sc.useSRP, 
        t_J2000, sc.x_idx_map);
    xdot = xdot + AttitudeDynamics(x, u, sc.num_MTBs, sc.num_RWs, sc.G_rw_b, sc.G_mtb_b, 
                                sc.I_rw, sc.I_sat, sc.MTB, t_J2000, sc.mass,  sc.Cd, sc.A, 
                                sc.CoPM, sc.useDT, sc.useGG, sc.x_idx_map, sc.u_idx_map);
    // [TODO:] Actuator Dynamics
    // [TODO:] Sensor Dynamics (bias, noise, etc)
    
    return xdot;
}

VectorXd OrbitalDynamics(const VectorXd& x, double mass, double Cd, double CR, double A, 
                                bool useDrag, bool useSRP, double t_J2000,
                                std::unordered_map<std::string, SliceDef> x_idx_map)
{
    VectorXd xdot = VectorXd::Zero(x.size());

    // Extract elements from state vector
    Vector3 r = x(x_idx_map["position"].to_seq());
    Vector3 v = x(x_idx_map["velocity"].to_seq());
    Quaternion q = vectorToQuaternion(x(x_idx_map["quaternion"].to_seq()));

    // Physics Models
    Vector3 vdot = gravitational_acceleration(r);
    
    if (useDrag){ 
        vdot = vdot + drag_acceleration(r, v, q, t_J2000, Cd, A, mass);
    }

    if (useSRP){
        vdot = vdot + SRP_acceleration(r ,q, t_J2000, CR, A, mass);
    }
    
    // Pack acceleration back into the state vector
    xdot(x_idx_map["position"].to_seq()) = v;
    xdot(x_idx_map["velocity"].to_seq()) = vdot;

    return xdot;
}

VectorXd AttitudeDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             const Eigen::MatrixXd& G_rw_b, const Eigen::MatrixXd& G_mtb_b,
                             double I_rw, const Matrix_3x3 I_sat, Magnetorquer MTB, double t_J2000,
                             double mass, double Cd, double A, const Vector3 CoPM,
                             bool useDT, bool useGG, std::unordered_map<std::string, SliceDef> x_idx_map,
                             std::unordered_map<std::string, SliceDef> u_idx_map)
{
    
    // Assert matrix sizes
    assert(u.size() == (num_MTBs + num_RWs+1)); // num_MTB + num_RWs torques + Jetson ON/OFF
    assert(G_rw_b.rows() == 3); // Orientation matrix has 3 element vectors
    assert(G_rw_b.cols() == num_RWs); // 1 column for each RW
    assert(G_mtb_b.rows() == 3); // 3D vector for each MTB
    assert(G_mtb_b.cols() == num_MTBs); // 1 column for each MTB

    VectorXd xdot = VectorXd::Zero(x.size());

    // Extract elements of state vector
    Vector3 r = x(x_idx_map["position"].to_seq());
    // Quaternion q{x(6), x(7), x(8), x(9)}; // initialize attitude quaternion
    Quaternion q = vectorToQuaternion(x(x_idx_map["quaternion"].to_seq()));
    q.normalize();
    Vector3 omega = x(x_idx_map["angular_rate"].to_seq());
    VectorXd omega_rw(num_RWs);
    omega_rw = x(x_idx_map["rw_speeds"].to_seq());

    Vector3 tau;
    
    /* Attitude Dynamics */
    Quaternion omega_quat {0, omega(0), omega(1), omega(2)};
    Quaternion qdot_quat = 0.5*q*omega_quat;
    Vector4 qdot{qdot_quat.w(), qdot_quat.x(), qdot_quat.y(), qdot_quat.z()};

    // Reaction Wheels
    auto h_rw = I_rw*omega_rw;
    u_idx_map["rw_torques"].to_seq();
    auto tau_rw = u(u_idx_map["rw_torques"].to_seq());
    tau = -G_rw_b*tau_rw;
    
    // Magnetorquers
    tau += MTB.getTorque(u(u_idx_map["mtb_torques"].to_seq()), q, MagneticField(r, t_J2000));
    // Vector3 tau_mtb = MTB.getTorque(u(Eigen::seqN(0,num_MTBs)), q, MagneticField(r, t_J2000));

    /* Perturbations */
    if (useDT) {
        Vector3 v = x(x_idx_map["velocity"].to_seq());
        tau += drag_torque(r, v, q, t_J2000, Cd, A, mass, CoPM);
    }
    if (useGG) {
        tau += gravity_gradient_torque(r, I_sat);
    }

    // Gyrostat Equation
    Vector3 h_sc = I_sat*omega + G_rw_b*h_rw;
    Vector3 omega_dot = I_sat.inverse()*(-omega.cross(h_sc) + tau);
    
    // Reaction wheel speeds
    auto omega_dot_rw = tau_rw/I_rw;

    // Pack into state derivative vector
    xdot(x_idx_map["quaternion"].to_seq()) = qdot;
    xdot(x_idx_map["angular_rate"].to_seq()) = omega_dot;
    xdot(x_idx_map["rw_speeds"].to_seq()) = omega_dot_rw;

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
    x_new(SC.x_idx_map["quaternion"].to_seq()) = x_new(SC.x_idx_map["quaternion"].to_seq())/x_new(SC.x_idx_map["quaternion"].to_seq()).norm();
    // sun position
    x_new(SC.x_idx_map["sun_position"].to_seq()) = sun_position_eci(t_J2000 + dt);
    // magnetic field 
    x_new(SC.x_idx_map["magnetic_field"].to_seq()) = MagneticField( x_new(SC.x_idx_map["position"].to_seq()), t_J2000 + dt);
    // bias 
    static std::normal_distribution<double> bias_noise_dist(0, SC.gyro_sigma_w);
    Vector3 bias_noise = Vector3::NullaryExpr([&](){return bias_noise_dist(gen);});
    x_new(SC.x_idx_map["gyro_bias"].to_seq()) = x_new(SC.x_idx_map["gyro_bias"].to_seq()) + dt*(bias_noise); // - bias/sc.gyro_correlation_time);
    // battery
    x_new(SC.x_idx_map["battery"].to_seq()) = x(SC.x_idx_map["battery"].to_seq()) + dt * PowerConsumptionWrapper(x, u, SC);
    // enforce SOC limit
    x_new(SC.x_idx_map["battery_soc"].to_idx()) = fmax(0,fmin(100, x_new(SC.x_idx_map["battery_soc"].to_idx())));

    return x_new;
}

VectorXd PowerConsumptionWrapper(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC) 
{
    return PowerConsumption(x, u, SC.resistances, SC.u_idx_map, SC.G_sp_b, SC.solar_panel_efficiency, 
                            SC.solar_panel_area, SC.x_idx_map, SC.battery_capacity, SC.battery_thermal_mass, 
                            SC.battery_radiative_loss, SC.solar_heat_factor, SC.max_pack_voltage, SC.battery_internal_resistance,
                            SC.mb_power, SC.num_MTBs, SC.num_RWs, SC.jetson_power);

}


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pyphysics, m) {
    m.doc() = "pybind11 physics plugin";   // module docstring    

    m.def("rk4", &rk4, "rk4 integrator");
    m.def("ECI2GEOD", &ECI2GEOD, "ECI to Geodetic Coordinates");
}
#endif