cmake_minimum_required(VERSION 3.12)
project(3dDynamicsSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("include/")

find_package(Eigen3 REQUIRED)

#library creation
add_library(${PROJECT_NAME} SHARED src/RigidBody.cpp src/utils.cpp) # Create shared library
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen )

#building the executable targets of this library
add_executable(3dDynamicsSim_main src/main.cpp)
target_link_libraries(3dDynamicsSim_main PRIVATE 3dDynamicsSim)

add_executable(SatelliteAroundEarth src/SatelliteAroundEarth.cpp)
target_link_libraries(SatelliteAroundEarth Eigen3::Eigen)

add_executable(SatelliteAroundEarthAndMoon src/SatelliteAroundEarthAndMoon.cpp)
target_link_libraries(SatelliteAroundEarthAndMoon Eigen3::Eigen)