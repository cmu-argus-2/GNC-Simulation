cmake_minimum_required(VERSION 3.12)
project(3dDynamicsSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O1")

include_directories("include/")

find_package(Eigen3 REQUIRED)
find_package(matplotlib_cpp REQUIRED)

add_compile_options(-fopt-info-missed=missed.all)

add_executable(SatelliteAroundEarth src/SatelliteAroundEarth.cpp)
target_link_libraries(SatelliteAroundEarth Eigen3::Eigen matplotlib_cpp::matplotlib_cpp)

add_executable(SatelliteAroundEarthAndMoon src/SatelliteAroundEarthAndMoon.cpp)
target_link_libraries(SatelliteAroundEarthAndMoon Eigen3::Eigen matplotlib_cpp::matplotlib_cpp)
