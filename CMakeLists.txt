cmake_minimum_required(VERSION 3.12)
project(3dDynamicsSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("include/")

find_package(Eigen3 REQUIRED)


add_compile_options(
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


#library creation
add_library(${PROJECT_NAME} SHARED src/RigidBody.cpp src/utils_and_transforms.cpp) # Create shared library
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen )

#building the executable targets of this library
add_executable(3dDynamicsSim_main src/main.cpp)
target_link_libraries(3dDynamicsSim_main PRIVATE 3dDynamicsSim)

add_executable(SatelliteAroundEarth src/SatelliteAroundEarth.cpp)
target_link_libraries(SatelliteAroundEarth Eigen3::Eigen)

add_executable(SatelliteAroundEarthAndMoon src/SatelliteAroundEarthAndMoon.cpp)
target_link_libraries(SatelliteAroundEarthAndMoon Eigen3::Eigen)