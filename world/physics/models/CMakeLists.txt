add_library(srp SHARED SRP.cpp)
target_link_libraries(srp PUBLIC Eigen3::Eigen tjLib spice)