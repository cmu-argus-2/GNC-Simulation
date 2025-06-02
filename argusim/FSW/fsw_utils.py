
import numpy as np
    

def j2000_to_unix_time(j2000):
    
    return j2000 + 946684800 + 11*3600 + 58*60 + 55 # 559889620.293


def unix_time_to_julian_day(unix_time):
    """Takes in a unix timestamp and returns the julian day"""
    return unix_time / 86400 + 2440587.5


def approx_sun_position_ECI(utime):
    """
    Formula taken from "Satellite Orbits: Models, Methods and Applications", Section 3.3.2, page 70, by Motenbruck and Gill

    Args:
        - utime: Unix timestamp

    Returns:
        - Sun pointing in Earth Centered Inertial (ECI) frame (km)
    """
    JD = unix_time_to_julian_day(utime)
    OplusW = 282.94  # Ω + ω
    T = (JD - 2451545.0) / 36525

    M = np.radians(357.5256 + 35999.049 * T)

    long = np.radians(OplusW + np.degrees(M) + (6892 / 3600) * np.sin(M) + (72 / 3600) * np.sin(2 * M))
    r_mag = (149.619 - 2.499 * np.cos(M) - 0.021 * np.cos(2 * M)) * 10**6

    epsilon = np.radians(23.43929111)
    r_vec = np.array([r_mag * np.cos(long), r_mag * np.sin(long) * np.cos(epsilon), r_mag * np.sin(long) * np.sin(epsilon)])

    return r_vec

def reset_array(input_array):
    for i in range(len(input_array)):
        input_array[i] = 0.0


def unix_time_to_years_since_2020(unix_time):
    twentytwenty = 1577854800  #  1/1/2020 unix timestamp
    t = (unix_time - twentytwenty) / 31557600  # seconds in a Julian astronomical year
    return t


def _igrf13_5(gh, unix_time, latitude_degrees, elongitude_degrees, r_norm_km, cl, sl, p, q):
    # reset the lists that are passed by reference
    reset_array(cl)
    reset_array(sl)
    reset_array(p)
    reset_array(q)

    # colatitude
    colat = 90 - latitude_degrees

    # Declaration of variables
    fn = 0
    gn = 0
    kmx = 0
    ll = 0
    nc = 0
    x = 0.0
    y = 0.0
    z = 0.0
    t = 0.0
    tc = 0.0

    t = unix_time_to_years_since_2020(unix_time)
    tc = 1.0

    ll = 0

    # nc = int(nmx * (nmx + 2))
    nc = 35

    # kmx = int((nmx + 1) * (nmx + 2) / 2)
    kmx = 21

    r = r_norm_km
    ct = np.cos(colat * np.pi / 180)
    st = np.sin(colat * np.pi / 180)
    cl[0] = np.cos(elongitude_degrees * np.pi / 180)
    sl[0] = np.sin(elongitude_degrees * np.pi / 180)
    Cd = 1.0
    sd = 0.0
    l = 1
    m = 1
    n = 0

    ratio = 6371.2 / r
    rr = ratio**2

    p[0] = 1.0
    p[2] = st
    q[0] = 0.0
    q[2] = ct

    for k in range(2, kmx + 1):
        if n < m:
            m = 0
            n = n + 1
            rr = rr * ratio
            fn = n
            gn = n - 1

        fm = m

        if m == n:
            if k != 3:
                one = np.sqrt(1 - 0.5 / fm)
                j = k - n - 1
                p[k - 1] = one * st * p[j - 1]
                q[k - 1] = one * (st * q[j - 1] + ct * p[j - 1])
                cl[m - 1] = cl[m - 2] * cl[0] - sl[m - 2] * sl[0]
                sl[m - 1] = sl[m - 2] * cl[0] + cl[m - 2] * sl[0]
        else:
            gmm = m**2
            one = np.sqrt(fn**2 - gmm)
            two = np.sqrt(gn**2 - gmm) / one
            three = (fn + gn) / one
            i = k - n
            j = i - n + 1
            p[k - 1] = three * ct * p[i - 1] - two * p[j - 1]
            q[k - 1] = three * (ct * q[i - 1] - st * p[i - 1]) - two * q[j - 1]

        lm = ll + l
        one = (tc * gh[lm - 1] + t * gh[lm + nc - 1]) * rr

        if m != 0:
            two = (tc * gh[lm] + t * gh[lm + nc]) * rr
            three = one * cl[m - 1] + two * sl[m - 1]
            x = x + three * q[k - 1]
            z = z - (fn + 1) * three * p[k - 1]

            if st != 0:
                y = y + (one * sl[m - 1] - two * cl[m - 1]) * fm * p[k - 1] / st
            else:
                y = y + (one * sl[m - 1] - two * cl[m - 1]) * q[k - 1] * ct

            l = l + 2

        else:
            x = x + one * q[k - 1]
            z = z - (fn + 1) * one * p[k - 1]
            l = l + 1

        m = m + 1

    one = x
    x = x * Cd + z * sd
    z = z * Cd - one * sd

    return np.array([x, y, z])


class _PARAMS:
    gh = [ # 2020
        -29404.8, -1450.9, 4652.5, -2499.6, 2982.0,
        -2991.6, 1677.0, -734.6, 1363.2, -2381.2,
        -82.1, 1236.2, 241.9, 525.7, -543.4,
        903.0, 809.5, 281.9, 86.3, -158.4,
        -309.4, 199.7, 48.0, -349.7, -234.3,
        363.2, 47.7, 187.8, 208.3, -140.7,
        -121.2, -151.2, 32.3, 13.5, 98.9,
        66.0, 65.5, -19.1, 72.9, 25.1,
        -121.5, 52.8, -36.2, -64.5, 13.5,
        8.9, -64.7, 68.1, 80.6, -76.7,
        -51.5, -8.2, -16.9, 56.5, 2.2,
        15.8, 23.5, 6.4, -2.2, -7.2,
        -27.2, 9.8, -1.8, 23.7, 9.7,
        8.4, -17.6, -15.3, -0.5, 12.8,
    ]
    cl = [0.0 for _ in range(5)]
    sl = [0.0 for _ in range(5)]
    p = [0.0 for _ in range(21)]
    q = [0.0 for _ in range(21)]


def igrf(unix_timestamp, latitude_degrees, elongitude_degrees, r_norm_km):
    """Returns the fifth order approximation from the IGRF-13 model.
    Only contains data from 2020, so it should only be accurate from 2020-2025.
    Args:
        - unix_timestamp: A unix timestamp.
        - latitude_degrees: Latitude in degrees (geocentric)
        - elongitude_degrees: Longitude in degrees (geocentric)
        - r_norm_km: Distance from the center of the earth (km)
    Returns:
        - [x, y, z] the magnetic field in nanotesla in (North, East, Down)
    """
    return _igrf13_5(
        _PARAMS.gh,
        unix_timestamp,
        latitude_degrees,
        elongitude_degrees,
        r_norm_km,
        _PARAMS.cl,
        _PARAMS.sl,
        _PARAMS.p,
        _PARAMS.q,
    )


def igrf_eci(unix_timestamp, r_eci):
    """Returns the fifth order approximation from the IGRF-13 model.
    Only contains data from 2020, so it should only be accurate from 2020-2025.
    IGRF-13 takes in geocentric coordinates, and outputs in NED (North, East, Down).
    We solve this by applying the following conversions: ECI->ECEF->GEOC=>[IGRF]=>NED->ECEF->ECI

    Args:
        - unix_timestamp: A unix timestamp.
        - r_eci: Earth Centered Interital frame position (km)
    Returns:
        - [x, y, z] the magnetic field in nanotesla in ECI (Earth Centered Inertial)
    """
    ecef_eci = eci_to_ecef(unix_timestamp)
    eci_ecef = ecef_eci.transpose()

    r_ecef = np.dot(ecef_eci, r_eci)
    long, lat, _ = convert_ecef_to_geoc(r_ecef)

    b_ned = igrf(unix_timestamp, (lat / np.pi) * 180, (long / np.pi) * 180, np.linalg.norm(r_eci))
    ecef_ned = ned_to_ecef(long, lat)
    return np.dot(eci_ecef, np.dot(ecef_ned, b_ned))


## frames code
J2000 = 946684800  # unix timestamp for the Julian date 2000-01-01
MJD_ZERO = 2400000.5  # Offset of Modified Julian Days representation with respect to Julian Days.
JD2000 = 2451545.0  # Reference epoch (J2000.0), Julian Date
MJD2000 = 51544.5  # MJD at J2000.0
PI2 = 6.283185307179586
EQUATORIAL_RADIUS = 6378.137  # Equatorial raduis of the Earth (km)


def mjd(utime):
    """Returns the Modified Julian Date (MJD) for a given unix timestamp."""
    return utime / 86400.0 + 40587


def rotZ(theta):
    """Returns the rotation matrix for a given angle around the z-axis.

    :param theta: Angle in radians.
    :type theta: float
    :returns: A 3x3 numpy array.
    """
    return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def ERA(utime):
    """Returns the the ERA (Earth Rotation Angle) at a certain unix time stamp.
    Inspired by SOFA's iauEra00 function.

    :param utime: A unix timestamp
    :type utime: int
    :returns: The Earth Rotation Angle in radians.
    """
    # Days since J2000.0.
    d = mjd(utime)
    days = d - MJD2000
    # Fractional part of T (days).
    f = d % 1.0
    # Earth rotation angle at this UT1.
    theta = (PI2 * (f + 1.2790572732640 + 0.00273781191135448 * days)) % (2 * PI2)

    return theta


def earth_rotation(utime):
    """Computes rotation matrix based on the Earth Rotation Angle (ERA) at a certain unix time stamp."""
    # Compute Earth rotation angle
    era = ERA(utime)
    # Rotate Matrix and return
    R = rotZ(era)
    return R


def eci_to_ecef(utime):
    """Returns the rotation matrix from ECI (Earth Centered Inertial) to ECEF (Earth Centered Earth Fixed).
    Applies correction for Earth-rotation.
    Based on SatelliteDynamic's rECItoECEF.

    :param utime: A unix timestamp
    :type utime: int
    :returns: A 3x3 numpy array.
    """
    # we may choose to add bias_precession_nutation and polar motion in the future
    # rc2i = bias_precession_nutation(epc)
    R = earth_rotation(utime)
    # rpm  = polar_motion(epc)

    # return rpm @ r @ rc2i
    return R


def ecef_to_eci(utime):
    """Returns the rotation matrix from ECEF (Earth Centered Earth Fixed) to ECI (Earth Centered Inertial).

    :param utime: A unix timestamp
    :type utime: int
    :returns: A 3x3 numpy array.
    """
    return eci_to_ecef(utime).transpose()


def ned_to_ecef(lon, lat):
    """Returns the rotation matrix for transforming coordinates in an earth-centered NED frame
    to coordinates in an ECEF frame.

    :param lon: Longitude in radians (geocentric).
    :type lon: float
    :param lat: Latitude in radians (geocentric).
    :param lat: float
    :returns: A 3x3 numpy array.
    """
    return np.array(
        [
            [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
            [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
            [np.cos(lat), 0.0, -np.sin(lat)],
        ]
    )


def convert_ecef_to_geoc(ecef, degrees=False):
    """Converts from ECEF (Earth Centered Earth Fixed) to geocentric coordinates.

    :param ecef: ECEF coordinates in km.
    :type ecef: numpy.array
    :param degrees: If True, returns the coordinates in degrees.
    :type degrees: bool, optional
    :returns: A 3x1 numpy arary containing the geocentric coordinates long, lat, alt (radians, radians, km)
    """
    x, y, z = ecef
    lat = np.arctan2(z, np.sqrt(x * x + y * y))
    lon = np.arctan2(y, x)
    alt = np.sqrt(x * x + y * y + z * z) - EQUATORIAL_RADIUS

    # Convert output to degrees
    if degrees:
        lat = lat * 180.0 / np.pi
        lon = lon * 180.0 / np.pi

    return np.array([lon, lat, alt])
