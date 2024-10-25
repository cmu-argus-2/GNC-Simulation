# Shared library creation
add_library(actuators SHARED Magnetorquer.cpp ReactionWheel.cpp)
target_link_libraries(actuators PUBLIC Eigen3::Eigen tjLib)
target_include_directories(actuators PUBLIC .)

