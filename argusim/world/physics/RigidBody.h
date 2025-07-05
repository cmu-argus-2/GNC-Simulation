#ifndef C___RIGIDBODY_H
#define C___RIGIDBODY_H

#include "math/EigenWrapper.h"
#include "ParameterParser.h"
#include "utils_and_transforms.h"

#include <random>

// Random Seed and Algorithm Definition
std::random_device rd;
std::mt19937 gen(rd());

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
    bool useDT;
    bool useGG;
};


VectorXd f(const VectorXd& x, const VectorXd& u, Simulation_Parameters sc, double t_J2000) ;

VectorXd OrbitalDynamics(const VectorXd& x, double mass, double Cd, double CR, double A, 
                                bool useDrag, bool useSRP, bool useSun, bool useMoon, double t_J2000,
                                std::unordered_map<std::string, SliceDef> x_idx_map);

VectorXd AttitudeDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             const Eigen::MatrixXd& G_rw_b, const Eigen::MatrixXd& G_mtb_b,
                             double I_rw, const Matrix_3x3 I_sat, Magnetorquer MTB, double t_J2000,
                             double mass, double Cd, double A, const Vector3 CoPM,
                             bool useDT, bool useGG, std::unordered_map<std::string, SliceDef> x_idx_map,
                             std::unordered_map<std::string, SliceDef> u_idx_map);

VectorXd ActuatorDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             double I_rw, Magnetorquer MTB, 
                             std::unordered_map<std::string, SliceDef> x_idx_map,
                             std::unordered_map<std::string, SliceDef> u_idx_map);

VectorXd rk4(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC, double t_J2000, double dt);

VectorXd PowerConsumptionWrapper(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC); 

#endif   // C___RIGIDBODY_H
