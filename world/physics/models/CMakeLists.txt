add_library(models SHARED gravity.cpp SRP.cpp drag.cpp MagneticField.cpp)
target_link_libraries(models PUBLIC math Eigen3::Eigen tjLib spice)
target_include_directories(models PUBLIC ".")

pybind11_add_module(pymodels SHARED gravity.cpp SRP.cpp drag.cpp MagneticField.cpp)
target_link_libraries(pymodels PUBLIC math Eigen3::Eigen tjLib spice)
target_include_directories(pymodels PUBLIC ".")
target_compile_definitions(pymodels PRIVATE USE_PYBIND_TO_COMPILE)
