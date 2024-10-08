add_library(models SHARED gravity.cpp SRP.cpp drag.cpp)# MagneticField.cpp)
target_link_libraries(models PUBLIC math Eigen3::Eigen tjLib spice)
target_include_directories(models PUBLIC ".")