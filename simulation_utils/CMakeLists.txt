add_library(sim_utils SHARED 
    ExpressionEvaluation.cpp
    ParameterParser.cpp 
    IO.cpp
)
target_link_libraries(sim_utils PUBLIC Eigen3::Eigen IMULib tjLib math)
target_include_directories(sim_utils PUBLIC .)


add_executable(SatelliteAroundEarth SatelliteAroundEarth.cpp)
target_link_libraries(SatelliteAroundEarth Eigen3::Eigen)

add_executable(SatelliteAroundEarthAndMoon SatelliteAroundEarthAndMoon.cpp)
target_link_libraries(SatelliteAroundEarthAndMoon Eigen3::Eigen)