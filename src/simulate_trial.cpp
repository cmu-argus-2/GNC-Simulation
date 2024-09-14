// sim includes
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "EigenWrapper.h"
#include "ParameterParser.h"
#include "RigidBody.h"
#include "tjLib/include/logging/MultiFileLogger.h"
#include "tjLib/include/math/random.h"
#include "tjLib/include/timing/utils.h"
#include "utils_and_transforms.h"

// defined in main.cpp
extern std::string trial_directory;

MultiFileLogger logger;

void simulate_trial(const Simulation_Parameters &params) {
    std::cout << "logging to " << trial_directory << std::endl;

    auto start = std::chrono::steady_clock::now();

    // initial state
    Vector3 init_pos_b_wrt_ECI_in_ECI{params.EARTH_RADIUS + params.init_altitute, 0, 0};   // [m]
    Quaternion init_ECI_q_b = Quaternion::Identity();
    Vector3 init_vel_b_wrt_ECI_in_ECI =
        params.orbital_velocity * params.orbital_plane_normal.cross(init_pos_b_wrt_ECI_in_ECI.normalized());   // [m/s]
    Vector3 init_vel_b_wrt_ECI_in_b   = init_ECI_q_b.inverse() * init_vel_b_wrt_ECI_in_ECI;                    // [m/s]
    Vector3 init_omega_b_wrt_ECI_in_b = DEG_2_RAD(3) * Vector3::UnitZ();   // [rad/s]

    // assert the initial position vector is orthogonal to the satellite's orbital plane normal vector
    double angle_between_pos_vector_and_orbital_plane_normal =
        acos(params.orbital_plane_normal.dot(init_pos_b_wrt_ECI_in_ECI.normalized()));
    assert(fabs(angle_between_pos_vector_and_orbital_plane_normal - M_PI_2) < 1e-10);

    RigidBody sat = RigidBody(params.satellite_mass, params.InertiaTensor, init_pos_b_wrt_ECI_in_ECI, init_ECI_q_b,
                              init_vel_b_wrt_ECI_in_b, init_omega_b_wrt_ECI_in_b);

    double t = 0.0;
    while (t <= params.MAX_TIME) {
        sat.clearAppliedForcesAndMoments();

        logger.log<3>(trial_directory + "/position_truth.bin", t, sat.get_pos_b_wrt_g_in_g(), "time [s]",
                      {"x [m]", "y [m]", "z [m]"});
        logger.log<3>(trial_directory + "/velocity_truth.bin", t, sat.get_vel_b_wrt_g_in_b(), "time [s]",
                      {"x [m/s]", "y [m/s]", "z [m/s]"});
        logger.log<3>(trial_directory + "/omega_truth.bin", t, RAD_2_DEG(sat.get_omega_b_wrt_g_in_b()), "time [s]",
                      {"x [deg/s]", "y [deg/s]", "z [deg/s]"});
        logger.log<3>(trial_directory + "/attitude_truth.bin", t,
                      RAD_2_DEG(intrinsic_zyx_decomposition(sat.get_g_q_b())), "time [s]",
                      {"yaw [deg]", "pitch [deg]", "roll [deg]"});

        // TODO(Pedro) apply your controller's forces and torques here
        // sat.applyForce(TODO, TODO);
        // sat.applyMoment(TODO);

        sat.rk4(params.dt);
        t += params.dt;
    }

    auto end = std::chrono::steady_clock::now();

    printElapsed(start, end);

    // ============= Print some one-off statistics to a yaml file =============
    std::string oneoff_stats_filename = trial_directory + "/oneoff_stats.yaml";
    std::ofstream file(oneoff_stats_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file \"" << oneoff_stats_filename << "\"for writing." << std::endl;
        return;
    }

    file << "Simulation Duration [s]: " << t << std::endl;
    file.close();
    // ========================================================================
}
