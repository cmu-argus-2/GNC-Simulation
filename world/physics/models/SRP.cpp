#include "SRP.h"

#include "SpiceUsr.h"
#include "math/EigenWrapper.h"

#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats

#ifdef print
#undef print   // we defined our own print funciton that clasehes with pybind's internal print; ignore ours
#endif
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

#pragma GCC diagnostic pop
#endif

// provide absolute path to ephemeris data file located at PROJECT_ROOT/data/de440.bsp
SRP::SRP(std::string ephemeris_data_filepath) {
    furnsh_c(ephemeris_data_filepath.c_str());
}

Vector3 SRP::sun_position_eci(double t_J2000) {
    SpiceDouble state[6];
    SpiceDouble lt;

    spkezr_c("sun", t_J2000, "J2000", "NONE", "earth", state, &lt);
    Vector3 sun_pos(1000.0 * state[0], 1000.0 * state[1],
                    1000.0 * state[2]);   // convert km to m and cast SpiceDouble into Vector3
    return sun_pos;
}

#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pymodels, m) {
    pybind11::class_<SRP>(m, "SRP").def(pybind11::init<std::string>()).def("sun_position_eci", &SRP::sun_position_eci);
}

#endif