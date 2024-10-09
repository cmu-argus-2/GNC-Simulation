# Shared library creation
add_library(actuators SHARED Magnetorquer.cpp ReactionWheel.cpp Motor.cpp)
target_link_libraries(actuators PUBLIC Eigen3::Eigen tjLib)

