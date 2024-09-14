cmake_minimum_required(VERSION 3.12)
project(3dDynamicsSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

include_directories("include/")

add_compile_options(
    -O3
    -g
    -pedantic               # Issue all the warnings demanded by strict ISO C
    -Wall                   # Warn about most questionable operations
    -Wextra                 # Even more warnings
    -Wwrite-strings         # Warn if not treating string literals as "const"
    -Wpointer-arith         # Warn about suspicious pointer operations
    -Wshadow                # Warn about variable shadowing
    -Wfloat-equal           # Warn about testing floats with ==
    -Werror
    -Wparentheses
    -Wunused-variable
    -Wunused-function
    -Wreturn-type
    -Wunused-value
    -Wswitch-default
    -Wuninitialized
    -Winit-self
    -Wno-comment            # Don't warn about /* in comments, it's common in the codebase and seems fine
    # -Wno-sign-compare       # Don't warn about sign compare; too many false positives even in external libraries
    # -Wno-unused-parameter   # Happens more frequently than you would think and it's generally ok
)

add_subdirectory(dependencies)

include_directories("dependencies/")

# Shared library creation
add_library(${PROJECT_NAME} SHARED 
    src/ExpressionEvaluation.cpp
    src/ParameterParser.cpp 
    src/RigidBody.cpp 
    src/simulate_trial.cpp
    src/utils_and_transforms.cpp
    src/IO.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen IMULib tjLib)

#building the executable targets of this library
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE 3dDynamicsSim)

add_executable(SatelliteAroundEarth src/SatelliteAroundEarth.cpp)
target_link_libraries(SatelliteAroundEarth Eigen3::Eigen)

add_executable(SatelliteAroundEarthAndMoon src/SatelliteAroundEarthAndMoon.cpp)
target_link_libraries(SatelliteAroundEarthAndMoon Eigen3::Eigen)