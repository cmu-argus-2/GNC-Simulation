add_library(sensors Sensors.cpp)
target_include_directories(sensors PUBLIC ".")
target_link_libraries(sensors tjLib math sim_utils models)
target_compile_options(sensors PRIVATE -fPIC)

pybind11_add_module(pysensors SHARED Sensors.cpp)
target_link_libraries(pysensors PUBLIC tjLib math sim_utils models)
target_include_directories(pysensors PUBLIC "sensors")
target_compile_definitions(pysensors PRIVATE USE_PYBIND_TO_COMPILE)