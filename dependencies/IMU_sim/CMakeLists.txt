add_library(IMULib IMU.cpp Sensor.cpp Bias.cpp IMUparams.cpp)
target_include_directories(IMULib PUBLIC ".")
target_link_libraries(IMULib tjLib)
target_compile_options(IMULib PRIVATE -fPIC)
