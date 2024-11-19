add_compile_options(-std=c++2a) # CMake 2.8.12 or newer
find_package (Eigen3 3.4 REQUIRED)
add_library(math SHARED math/utils_and_transforms.cpp)
target_link_libraries(math PUBLIC Eigen3::Eigen tjLib spice)
target_include_directories(math PUBLIC "math")

add_subdirectory(physics/models)

add_library(physics SHARED physics/RigidBody.cpp)
target_link_libraries(physics PUBLIC Eigen3::Eigen tjLib models math sim_utils actuators)
target_include_directories(physics PUBLIC "physics")

pybind11_add_module(pyphysics SHARED physics/RigidBody.cpp)
target_link_libraries(pyphysics PUBLIC Eigen3::Eigen tjLib models math sim_utils actuators)
target_include_directories(pyphysics PUBLIC "physics")
target_compile_definitions(pyphysics PRIVATE USE_PYBIND_TO_COMPILE)
