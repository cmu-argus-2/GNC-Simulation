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
 * @param mass : mass of the spacecraft [UNITS: Kg]
 * @param J_sat : inertia matrix of the spacecraft [UNITS: kg*m^2]
 * @param A : Reference area of the spacecraft [UNITS: m^2]
 * @param Cd : Drag coefficient of the spacecraft [UNITS: unitless]
 * @param CR : Reflectivity coefficient of the spacecraft [UNITS: unitless]
 * @param num_RWs : number of reaction wheels
 * @param G_rw_b : orientation matrix for reaction wheels (columns are unit vectors of each RW in body frame)
 * @param I_rw : inertia of each reaction wheel about its rotary axis [UNITS: kg*m^2]
 * @param num_MTBs : number of magnetorquers
 * @param G_mtb_b : orientation matrix for magnetorquers (columns are unit vectors of each MTB in body frame)
 * @param useDrag : Flag to enable/disable atmospheric drag perturbation
 * @param useSRP : Flag to enable/disable solar radiation pressure perturbation
 * @param useDT : Flag to enable/disable drag torque perturbation
 * @param useGG : Flag to enable/disable gravity gradient torque perturbation
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

/**
 * @brief Computes the time derivative of the state vector
 * 
 * @param x : state vector
 * @param u : control input vector
 * @param sc : Instance of Simulation_Parameters class holding physical properties and simulation settings
 * @param t_J2000 : current time in seconds since J2000 epoch
 * @return time derivative of the state vector
 */
VectorXd f(const VectorXd& x, const VectorXd& u, Simulation_Parameters sc, double t_J2000) ;

/**
 * @brief Computes the translational orbital dynamics portion of the state derivative
 * 
 * @param x : state vector
 * @param mass : mass of the spacecraft [UNITS: Kg]
 * @param Cd : Drag coefficient of the spacecraft [UNITS: unitless]
 * @param CR : Reflectivity coefficient of the spacecraft [UNITS: unitless]
 * @param A : Reference area of the spacecraft [UNITS: m^2]
 * @param useDrag : Flag to enable/disable atmospheric drag perturbation
 * @param useSRP : Flag to enable/disable solar radiation pressure perturbation
 * @param useSun : Flag to enable/disable solar gravity perturbation
 * @param useMoon : Flag to enable/disable lunar gravity perturbation
 * @param t_J2000 : current time in seconds since J2000 epoch
 * @param x_idx_map : map of state indices
 * @return time derivative of the orbital state portion of the state vector
 */
VectorXd OrbitalDynamics(const VectorXd& x, double mass, double Cd, double CR, double A, 
                                bool useDrag, bool useSRP, bool useSun, bool useMoon, double t_J2000,
                                std::unordered_map<std::string, SliceDef> x_idx_map);

/**
 * @brief Computes the rotational attitude dynamics portion of the state derivative
 * 
 * @param x : state vector
 * @param u : control input vector
 * @param num_MTBs : number of magnetorquers
 * @param num_RWs : number of reaction wheels
 * @param G_rw_b : orientation matrix for reaction wheels (columns are unit vectors of each RW in body frame)
 * @param G_mtb_b : orientation matrix for magnetorquers (columns are unit vectors of each MTB in body frame)
 * @param I_rw : inertia of each reaction wheel about its rotary axis [UNITS: kg*m^2]
 * @param I_sat : inertia matrix of the spacecraft [UNITS: kg*m^2]
 * @param MTB : Instance of Magnetorquer class
 * @param t_J2000 : current time in seconds since J2000 epoch
 * @param mass : mass of the spacecraft [UNITS: Kg]
 * @param Cd : Drag coefficient of the spacecraft [UNITS: unitless]
 * @param A : Reference area of the spacecraft [UNITS: m^2]
 * @param CoPM : Center of Pressure to Center of Mass vector [UNITS: m]
 * @param useDT : Flag to enable/disable drag torque perturbation
 * @param useGG : Flag to enable/disable gravity gradient torque perturbation
 * @param x_idx_map : map of state indices
 * @param u_idx_map : map of control input indices
 * @return time derivative of the attitude state portion of the state vector
 */
VectorXd AttitudeDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             const Eigen::MatrixXd& G_rw_b, const Eigen::MatrixXd& G_mtb_b,
                             double I_rw, const Matrix_3x3 I_sat, Magnetorquer MTB, double t_J2000,
                             double mass, double Cd, double A, const Vector3 CoPM,
                             bool useDT, bool useGG, std::unordered_map<std::string, SliceDef> x_idx_map,
                             std::unordered_map<std::string, SliceDef> u_idx_map);

/**
 * @brief Computes the actuator dynamics portion of the state derivative
 * 
 * @param x : state vector
 * @param u : control input vector
 * @param num_MTBs : number of magnetorquers
 * @param num_RWs : number of reaction wheels
 * @param I_rw : inertia of each reaction wheel about its rotary axis [UNITS: kg*m^2]
 * @param MTB : Instance of Magnetorquer class
 * @param x_idx_map : map of state indices
 * @param u_idx_map : map of control input indices
 * @return time derivative of the actuator state portion of the state vector
 */
VectorXd ActuatorDynamics(const VectorXd& x, const VectorXd& u,int num_MTBs, int num_RWs, 
                             double I_rw, Magnetorquer MTB, 
                             std::unordered_map<std::string, SliceDef> x_idx_map,
                             std::unordered_map<std::string, SliceDef> u_idx_map);

/**
 * @brief 4th-order Runge-Kutta integrator for state propagation
 * 
 * @param x : state vector at current time step
 * @param u : control input vector at current time step
 * @param SC : Instance of Simulation_Parameters class holding physical properties and simulation settings
 * @param t_J2000 : current time in seconds since J2000 epoch
 * @param dt : time step for integration
 * @return state vector at next time step
 */
VectorXd rk4(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC, double t_J2000, double dt);

/**
 * @brief: Wrapper for PowerDynamics function
 * 
 * @param x : state vector
 * @param u : control input vector
 * @param SC : Instance of Simulation_Parameters class holding physical properties and simulation settings
 * @return time derivative of the power state portion of the state vector
 */
VectorXd PowerDynamicsWrapper(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC); 

/**
 * @brief: Wrapper for Battery function
 * 
 * @param x : state vector
 * @param u : control input vector
 * @param SC : Instance of Simulation_Parameters class holding physical properties and simulation settings
 * @return time derivative of the battery state portion of the state vector
 */
VectorXd BatteryWrapper(const VectorXd& x, const VectorXd& u, Simulation_Parameters SC); 


#endif   // C___RIGIDBODY_H
