#include "utils_and_transforms.h"

#include "SpiceUsr.h"

// rotation matrix elements under this threshhold will be reset to 0
static constexpr double ROT_MAT_0_THRESH = 1e-10;

// seconds elapsed between the Unix and J2000 epoch
static constexpr int64_t J2000epochInUnixTime = 946727936;

int64_t unixToJ2000(int64_t unixSeconds) {
    return unixSeconds - J2000epochInUnixTime;
}

Matrix_3x3 toSkew(const Vector3& v) {
    Matrix_3x3 v_cross;
    v_cross << 0, -v(2), v(1),   //
        v(2), 0, -v(0),          //
        -v(1), v(0), 0;
    return v_cross;
}

Matrix_3x3 cleanRotMatrix(Matrix_3x3 R) {
    Matrix_3x3 R_cleaned = R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (std::fabs(R_cleaned(i, j)) < ROT_MAT_0_THRESH) {
                R_cleaned(i, j) = 0;
            }
        }
    }
    return R_cleaned;
}

Matrix_3x3 get_ECEF_R_ENU(double latitude_deg, double longitude_deg) {
    double latitude_rad  = DEG_2_RAD(latitude_deg);
    double longitude_rad = DEG_2_RAD(longitude_deg);
    Matrix_3x3 ECEF_R_ENU;
    ECEF_R_ENU << -sin(longitude_rad), -sin(latitude_rad) * cos(longitude_rad), cos(latitude_rad) * cos(longitude_rad),
        cos(longitude_rad), -sin(latitude_rad) * sin(longitude_rad), cos(latitude_rad) * sin(longitude_rad), 0,
        cos(latitude_rad), sin(latitude_rad);
    return ECEF_R_ENU;
}

Vector3 intrinsic_xyz_decomposition(const Quaternion& q) {
    Matrix_3x3 R = q.toRotationMatrix();
    double pitch = asin(R(0, 2));
    double yaw   = atan2(-R(0, 1), R(0, 0));
    double roll  = atan2(-R(1, 2), R(2, 2));
    return Vector3{yaw, pitch, roll};
}

Vector3 intrinsic_zyx_decomposition(const Quaternion& q) {
    Matrix_3x3 R = q.toRotationMatrix();
    double yaw   = atan2(R(1, 0), R(0, 0));
    double roll  = atan2(R(2, 1), R(2, 2));
    double pitch = asin(-R(2, 0));

    return Vector3{yaw, pitch, roll};
}

/* COORDINATE TRANSFORMATIONS FROM SPICE */

// Basic Utility functions
void loadAllKernels() {
    std::string sol_system_pos = "/home/kmkar/Documents/CMU/F24/Spacecraft_Lab/simulations/cpp_dynamics_sim/data/de440.bsp";
    std::string earth_rotation_pck = "/home/kmkar/Documents/CMU/F24/Spacecraft_Lab/simulations/cpp_dynamics_sim/data/earth_latest_high_prec.bpc";
    
    SpiceInt count;
    ktotal_c("ALL", &count);
    printf("Count %d\n", count);

    if (count == 0) {
        furnsh_c(sol_system_pos.c_str());
        furnsh_c(earth_rotation_pck.c_str());
    }; // only load kernel if not already loaded
    
}

// Convert CSPICE Double array to 3x3 Eigen Matrix
Matrix_3x3 Cspice2Eigen(SpiceDouble M[3][3]) {
    Matrix_3x3 R;
    R << M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2];
    return R;
}

// TRANSFORMS

Matrix_3x3 ECI2ECEF(double t_J2000) {
    SpiceDouble Rot[3][3];

    loadAllKernels();
    pxform_c("J2000", "ITRF93", t_J2000, Rot);
    
    return Cspice2Eigen(Rot);
}

Matrix_3x3 ECEF2ECI(double t_J2000) {
    SpiceDouble Rot[3][3];

    loadAllKernels();
    pxform_c("ITRF93", "J2000", t_J2000, Rot);
    
    return Cspice2Eigen(Rot);
}