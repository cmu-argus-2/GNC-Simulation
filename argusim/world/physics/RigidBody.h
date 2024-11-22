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
                             double I_rw, const Matrix_3x3 J_sat, Magnetorquer MTB, double t_J2000);

VectorXd rk4(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC, double t_J2000, double dt);


#endif   // C___RIGIDBODY_H
