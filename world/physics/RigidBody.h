#ifndef C___RIGIDBODY_H
#define C___RIGIDBODY_H

#include "math/EigenWrapper.h"
#include "ParameterParser.h"

/**
 * @brief Spacecraft struct
 *
 */
struct Spacecraft {
    // physical properties
    double mass;
    Matrix_3x3 J_sat;
    double A;

    // Drag and SRP properties
    double Cd;
    double CR;

    // Reaction Wheels
    int num_RWs;
    Eigen::MatrixXd G_rw_b;
    double I_rw;

    // Magnetorquers
    int num_MTBs;
    Eigen::MatrixXd G_mtb_b;

    // Physics Models
    bool useDrag;
    bool useSRP;
};


VectorXd f(const VectorXd& x, const VectorXd& u, Simulation_Parameters sc, double t_J2000) ;

VectorXd OrbitalDynamics(const VectorXd& x, double mass, double Cd, double CR, double A, 
                                bool useDrag, bool useSRP, double t_J2000);

VectorXd AttitudeDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             const Eigen::MatrixXd& G_rw_b, const Eigen::MatrixXd& G_mtb_b,
                             double I_rw, const Matrix_3x3 J_sat);

VectorXd rk4(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC, double t_J2000, double dt);

/* UTILITY FUNCTIONS*/
void set_pos_in_state(VectorXd &x, const Vector3 &pos);
void set_vel_in_state(VectorXd &x, const Vector3 &vel);
void set_quaternion_in_state(VectorXd &x, const Quaternion &quat);
void set_omega_in_state(VectorXd &x, const Vector3 &omega);
void set_omegaRW_in_state(VectorXd &x, const Vector3 &omegaRW);
Vector3 get_pos_from_state(VectorXd &x);
Vector3 get_vel_from_state(VectorXd &x);
Quaternion get_quaternion_from_state(VectorXd &x);
Vector3 get_omega_from_state(VectorXd &x);
VectorXd get_omegaRW_from_state(VectorXd &x);


#endif   // C___RIGIDBODY_H
