add_library(math SHARED math/utils_and_transforms.cpp)
target_link_libraries(math PUBLIC Eigen3::Eigen tjLib)
target_include_directories(math PUBLIC "math")

add_library(physics SHARED physics/RigidBody.cpp)
target_link_libraries(physics PUBLIC Eigen3::Eigen tjLib)
target_include_directories(physics PUBLIC "physics")
