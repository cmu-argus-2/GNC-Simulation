#include "gravity.h"
#include "drag.h"
#include "SRP.h"
#include "MagneticField.h"


#ifdef USE_PYBIND_TO_COMPILE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"   // purposely comparing floats
#include "pybind11/eigen.h"
#pragma GCC diagnostic pop
#endif


#ifdef USE_PYBIND_TO_COMPILE
PYBIND11_MODULE(pymodels, m) {
    m.doc() = "pybind11 model unit testing plugin";   // module docstring

    // Gravity functions
    m.def("grav_acc", &gravitational_acceleration, "gravitational_acceleration");
    m.def("grav_sph_acc", &spherical_acceleration, "gravitational acceleration due to spherical Earth");
    m.def("grav_J2_acc", &J2_perturbation, "gravitational acceleration due to J2");

    //Drag Functions
    m.def("drag_acc", &drag_acceleration, "drag acceleration");
    m.def("density", &density, "Harris-priester density");
    m.def("frontal_area", &FrontalAreaFactor, "Frontal Area factor");

    //SRP functions
    m.def("srp_acc", &SRP_acceleration, "SRP acceleration");
    m.def("sun_pos", &sun_position_eci, "ECI sun position");
    
    // Magnetic Field
    m.def("magnetic_field", &MagneticField, "magnetic field ECI");
    m.def("magnetic_field_sez", &MagneticFieldSEZ, "magnetic field SEZ");
    
}
#endif