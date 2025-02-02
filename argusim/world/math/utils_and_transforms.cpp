#include "utils_and_transforms.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <filesystem>

#include "SpiceUsr.h"
#include <cmath>
#include <random>

// rotation matrix elements under this threshhold will be reset to 0
static constexpr double ROT_MAT_0_THRESH = 1e-10;

// seconds elapsed between the Unix and J2000 epoch
static constexpr int64_t J2000epochInUnixTime = 946727936;

// Astronomical Unit [m]
static constexpr double ASTRONOMICAL_UNIT = 149597870700;

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


Matrix_3x3 random_SO3_rotation(std::normal_distribution<double> dist, std::mt19937 gen)
{
    Vector3 noise = Vector3::NullaryExpr([&](){return dist(gen);});

    Matrix_3x3 W = toSkew(noise);
    Matrix_3x3 W_so3 = W.exp();

    return W_so3;
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
    std::filesystem::path path(__FILE__);
    std::string root = path.parent_path().parent_path().parent_path().string(); // utils_and_transforms.cpp --> math --> world --> dynamics sim
    std::string data_folder = root + "/data/";


    std::string sol_system_spk = data_folder + "de440.bsp";
    std::string earth_rotation_pck = data_folder + "earth_latest_high_prec.bpc";
    std::string earth_dimensions_pck = data_folder + "pck00011.tpc";
    std::string leap_seconds_lsk = data_folder + "pck00011.tpc";
    std::string leap_seconds_lsk2 = data_folder + "naif0012.tls";
    
    SpiceInt count;
    ktotal_c("ALL", &count);

    if (count == 0) {
        furnsh_c(sol_system_spk.c_str());
        furnsh_c(earth_rotation_pck.c_str());
        furnsh_c(earth_dimensions_pck.c_str());
        furnsh_c(leap_seconds_lsk.c_str());
        furnsh_c(leap_seconds_lsk2.c_str());
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

Vector3 ECEF2GEOD(Vector3 v_ecef) {

    SpiceInt n;
    SpiceDouble radii[3];
    SpiceDouble v[3]; //ECEF vector as a spice double
    SpiceDouble f;

    SpiceDouble lon, lat, alt;

    loadAllKernels();
    
    bodvrd_c( "EARTH", "RADII", 3, &n, radii ); // extract earth radius and flattening info
    f = ( radii[0] - radii[2] ) / radii[0]; // flattening coeffficient

    vpack_c(v_ecef(0), v_ecef(1), v_ecef(2), v); // cast Vector 3 to SpiceDouble[3]
    recgeo_c(v, 1000*radii[0], f, &lon, &lat, &alt); //convert to lon-lat-alt //NOTE: radii is populated in km

    Vector3 geod (lon, lat, alt);
    
    return geod;
}

Vector3 ECI2GEOD(Vector3 v_eci, double t_J2000){
    Vector3 v_ecef = ECI2ECEF(t_J2000)*v_eci;
    Vector3 r_geod = ECEF2GEOD(v_ecef);

    r_geod(0) = RAD_2_DEG(r_geod(0));
    r_geod(1) = RAD_2_DEG(r_geod(1));
    
    return r_geod;
}

Vector3 SEZ2ECEF(Vector3 r_sez, double latitude, double longitude)
{
    Vector3 r_ecef;
    Matrix_3x3 R {{sin(latitude)*cos(longitude), -sin(longitude), cos(latitude)*cos(longitude)},
                  {sin(latitude)*sin(longitude), cos(longitude), cos(latitude)*sin(longitude)},
                  {-cos(latitude), 0.0, sin(latitude)}};

    r_ecef = R*r_sez;
    return r_ecef;
}

Vector6 KOE2ECI(Vector6 KOE, double t_J2000)
{
    loadAllKernels();

    SpiceDouble conics[8];
    
    conics[0] = KOE(0)*(1 - KOE(1))/1000.0;
    conics[1] = KOE(1);
    conics[2] = DEG_2_RAD(KOE(2));
    conics[3] = DEG_2_RAD(KOE(3));
    conics[4] = DEG_2_RAD(KOE(4));

    double f = DEG_2_RAD(KOE(5));
    conics[5] = atan2(-sqrt(1-pow(KOE(1),2))*sin(f), -KOE(1)-cos(f)) + M_PI - 
                KOE(1)*(sqrt(1-pow(KOE(1),2))*sin(f))/(1 + KOE(1)*cos(f)); // True anomaly to mean anomaly calculation

    conics[6] = t_J2000;
    conics[7] = 3.98600435507e5; // km^3/s^2

    SpiceDouble s[6];
    conics_c(conics, t_J2000, s);

    Vector6 state;
    state << s[0], s[1], s[2], s[3], s[4], s[5];
    state = state*1000; // convert km, km/s to m and m/s
    return state;
}


/* TIME UTILITIES */
Vector5 TJ2000toUTC(double t_J2000)
{
    loadAllKernels();
    const int oplen = 35;
    SpiceChar utc_datestring[oplen];
    Vector5 utc_date;

    et2utc_c(t_J2000, "ISOD", 3, oplen, utc_datestring);
    std::string datestring = utc_datestring;
    utc_date(0) = atof(datestring.substr(0,4).c_str());
    utc_date(1) = atof(datestring.substr(5,3).c_str());
    utc_date(2) = atof(datestring.substr(9,2).c_str());
    utc_date(3) = atof(datestring.substr(12,2).c_str());
    utc_date(4) = atof(datestring.substr(15,6).c_str());

    return utc_date;
}

std::string TJ2000toUTCString(double t_J2000)
{
    loadAllKernels();
    const int oplen = 35;
    SpiceChar utc_datestring[oplen];

    et2utc_c(t_J2000, "ISOC", 3, oplen, utc_datestring);
    std::string datestring = utc_datestring;

    return datestring;
}

double UTCStringtoTJ2000 (std::string UTC)
{
    loadAllKernels();

    std::replace(UTC.begin(), UTC.end(), ' ', 'T'); // replaces space in string with a T to follow ISO format
    const char * UTC_char = UTC.c_str();

    double t_J2000;
    utc2et_c(UTC_char, &t_J2000);

    return t_J2000;
}

double UTCStringtoHours(std::string UTC)
{
    int hours, minutes, seconds;
    char colon;
    std::istringstream iss(UTC);
    iss >> hours >> colon >> minutes >> colon >> seconds;
    double t_h = hours + minutes / 60.0 + seconds / 3600.0;
    return t_h;
}

Eigen::Vector3d sun_position_mod(double t_J2000)
{
    // Number of Julian centuries from J2000 epoch.
    // double t_J2000 = (jd_tdb - JD_J2000) / 36525.0;
    t_J2000 = t_J2000 / (36525.0 * 86400.0);

    // Mean anomaly of the Sun [deg].
    double Ms = 357.5291092 + 35999.05034 * t_J2000;

    // Convert Ms to [rad] and limit to the interval [0, 2π].
    Ms = fmod(Ms * M_PI / 180.0, 2 * M_PI);

    // Compute auxiliary variables.
    double sin_Ms = sin(Ms);
    double cos_Ms = cos(Ms);

    // Let's compute `sin(2Ms)` and `cos(2Ms)` more efficiently.
    double sin_2Ms = 2 * sin_Ms * cos_Ms;
    double cos_2Ms = cos_Ms * cos_Ms - sin_Ms * sin_Ms;

    // Mean longitude of the Sun [deg].
    double lambda_m = 280.460 + 36000.771 * t_J2000;

    // Ecliptic latitude of the Sun [deg].
    double lambda_e = lambda_m + 1.914666471 * sin_Ms + 0.019994643 * sin_2Ms;

    // Obliquity of the ecliptic [deg].
    double epsilon = 23.439291 - 0.0130042 * t_J2000;

    // Convert lambda_e and epsilon to [rad] and limit to the interval [0, 2π].
    lambda_e = fmod(lambda_e * M_PI / 180.0, 2 * M_PI);
    epsilon = fmod(epsilon * M_PI / 180.0, 2 * M_PI);

    // Auxiliary variables.
    double sin_epsilon = sin(epsilon);
    double cos_epsilon = cos(epsilon);
    double sin_lambda_e = sin(lambda_e);
    double cos_lambda_e = cos(lambda_e);

    // Distance of the Sun from Earth [m].
    double r = (1.000140612 - 0.016708617 * cos_Ms - 0.000139589 * cos_2Ms) * ASTRONOMICAL_UNIT;

    // Compute the Sun vector represented in the Mean Equinox of Date (MOD).
    Eigen::Vector3d s_mod;
    s_mod << r * cos_lambda_e, r * cos_epsilon * sin_lambda_e, r * sin_epsilon * sin_lambda_e;

    return s_mod;
}

double equation_of_time(double t_J2000)
{
    // Mean longitude of the Sun.
    double lambda_m = fmod(280.460 + 36000.771 * t_J2000, 360);

    // Mean anomaly of the Sun.
    double Ms = fmod(357.5291092 + 35999.05034 * t_J2000, 360) * M_PI / 180.0;

    // Auxiliary variables.
    double sin_Ms = sin(Ms);
    double sin_2Ms = sin(2 * Ms);

    // Ecliptic latitude of the Sun.
    double lambda_ecliptic = fmod(lambda_m + 1.914666471 * sin_Ms + 0.019994643 * sin_2Ms, 360) * M_PI / 180.0;

    // Compute the equation of time [deg].
    double eot = -1.914666471 * sin_Ms - 0.019994643 * sin_2Ms + 2.466 * sin(2 * lambda_ecliptic) - 0.0053 * sin(4 * lambda_ecliptic);

    return eot * M_PI / 180.0;
}

double LTAN_to_RAAN(double ltan, double t_J2000)
{
    // Get the sun position at noon (UT) represented in the MOD reference frame.
    Eigen::Vector3d s_i = sun_position_mod(t_J2000);

    // Get the desired angle between the Sun and the ascending node [rad].
    double alpha = (ltan - 12) * M_PI / 12;

    // Get the right ascension of the Sun in the MOD reference frame. This is the Sun
    // apparent local time.
    double salt = atan2(s_i[1], s_i[0]);

    // Get the equation of time to compute the Sun mean local time [rad].
    double eot = equation_of_time(t_J2000);

    // Compute the Sun mean local time.
    double smlt = salt + eot;

    // Compute the desired RAAN in the interval [0, 2π].
    double raan = fmod(smlt + alpha, 2 * M_PI);
    if (raan < 0) {
        raan += 2 * M_PI;
    }

    return raan * 180.0 / M_PI;
}

double LTDN_to_RAAN(double ltdn, double t_J2000) {
    return LTAN_to_RAAN(fmod(ltdn + 12, 24), t_J2000);
}
