# Shared library creation
add_library(actuators SHARED Magnetorquer.cpp ReactionWheel.cpp)
target_link_libraries(actuators PUBLIC Eigen3::Eigen tjLib)
target_include_directories(actuators PUBLIC .)

pybind11_add_module(pymagnetorquers SHARED Magnetorquer.cpp)
target_link_libraries(pymagnetorquers PUBLIC Eigen3::Eigen tjLib)
target_include_directories(pymagnetorquers PUBLIC "magnetorquers")
target_compile_definitions(pymagnetorquers PRIVATE USE_PYBIND_TO_COMPILE)
