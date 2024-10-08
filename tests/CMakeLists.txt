include_directories(../world/physics/models)

pybind11_add_module(pymodels SHARED binder.cpp)
target_link_libraries(pymodels PUBLIC Eigen3::Eigen tjLib models math spice)
target_include_directories(pymodels PUBLIC "physics")
target_compile_definitions(pymodels PRIVATE USE_PYBIND_TO_COMPILE)
