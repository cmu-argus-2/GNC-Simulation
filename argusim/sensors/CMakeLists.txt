add_library(deployables Deployable.cpp)
target_include_directories(deployables PUBLIC ".")
target_link_libraries(deployables Eigen3::Eigen tjLib math)
target_compile_options(deployables PRIVATE -fPIC)

pybind11_add_module(pydeployables SHARED Deployable.cpp)
target_link_libraries(pydeployables PUBLIC Eigen3::Eigen tjLib)
target_include_directories(pydeployables PUBLIC "deployables")
target_compile_definitions(pydeployables PRIVATE USE_PYBIND_TO_COMPILE)

add_library(sensors Sensors.cpp)
target_include_directories(sensors PUBLIC ".")
target_link_libraries(sensors tjLib math sim_utils models deployables)
target_compile_options(sensors PRIVATE -fPIC)

pybind11_add_module(pysensors SHARED Sensors.cpp)
target_link_libraries(pysensors PUBLIC tjLib math sim_utils models deployables)
target_include_directories(pysensors PUBLIC "sensors")
target_compile_definitions(pysensors PRIVATE USE_PYBIND_TO_COMPILE)