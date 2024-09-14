#include <chrono>
#include <iostream>

#include "RigidBody.h"
#include "utils_and_transforms.h"

// TODO(tushaar): move me to a constants file
constexpr double EARTH_RADIUS = 63'781'370;   // [m]
constexpr double mass         = 1.5;          //[kg]

int main() {
    auto start = std::chrono::steady_clock::now();

    double maxTime  = 2 * 90 * 60;   // [s]
    double currTime = 0;             // [s]
    double dt       = 0.02;          // [s]

    // Right Ascension of the Ascending Node
    double RAAN_deg                    = 0;      // [deg]
    double orbital_incliniation_deg    = 98.0;   // [deg]
    double init_altitute_km            = 600;    // [km]
    double orbital_velocity_km_per_sec = 7.56;   // [km/s]

    double RAAN                 = DEG_2_RAD(RAAN_deg);                   // [m]
    double orbital_incliniation = DEG_2_RAD(orbital_incliniation_deg);   // [rad]
    double init_altitute        = 1000 * init_altitute_km;               // [m]
    double orbital_velocity     = 1000 * orbital_velocity_km_per_sec;    // [m/s]

    // TODO(tushaar): verify ME
    Vector3 orbital_plane_normal{sin(orbital_incliniation) * sin(RAAN),    //
                                 -sin(orbital_incliniation) * cos(RAAN),   //
                                 cos(orbital_incliniation)};

    // initial state
    Vector3 init_pos_b_wrt_ECI_in_ECI{EARTH_RADIUS + init_altitute, 0, 0};   // [m]
    Quaternion init_ECI_q_b = Quaternion::Identity();
    Vector3 init_vel_b_wrt_ECI_in_ECI =
        orbital_velocity * orbital_plane_normal.cross(init_pos_b_wrt_ECI_in_ECI.normalized());   // [m/s]
    Vector3 init_vel_b_wrt_ECI_in_b   = init_ECI_q_b.inverse() * init_vel_b_wrt_ECI_in_ECI;      // [m/s]
    Vector3 init_omega_b_wrt_ECI_in_b = DEG_2_RAD(3) * Vector3::UnitZ();                         // [rad/s]

    // assert the initial position vector is orthogonal to the satellite's orbital plane normal vector
    double angle_between_pos_vector_and_orbital_plane_normal =
        acos(orbital_plane_normal.dot(init_pos_b_wrt_ECI_in_ECI.normalized()));
    assert(fabs(angle_between_pos_vector_and_orbital_plane_normal - M_PI_2) < 1e-10);

    Matrix_3x3 InertiaTensor;
    InertiaTensor << 2e-3, 0, 0, 0, 2e-3, 0, 0, 0, 3e-3;

    RigidBody rigidBody1 = RigidBody(mass, InertiaTensor, init_pos_b_wrt_ECI_in_ECI, init_ECI_q_b,
                                     init_vel_b_wrt_ECI_in_b, init_omega_b_wrt_ECI_in_b);

    while (currTime <= maxTime) {
        rigidBody1.clearAppliedForcesAndMoments();
        // TODO(Pedro) apply your controller's forces and torques here
        // rigidBody1.applyForce(TODO, TODO);
        // rigidBody1.applyMoment(TODO);

        rigidBody1.update(dt);
        currTime += dt;
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time to Simulate: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    return 0;
}