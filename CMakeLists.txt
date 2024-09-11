cmake_minimum_required(VERSION 3.12)
project(3dDynamicsSim LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories("include/")

find_package(Eigen3 REQUIRED)
find_package(matplotlib_cpp REQUIRED)


#library creation
include(GNUInstallDirs)
add_library(${PROJECT_NAME} SHARED src/RigidBody.cpp src/utils.cpp) # Create shared library
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen matplotlib_cpp::matplotlib_cpp)

#Installing the library's header files to the place CMAKE usually looks for header files
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

#Installing the shared object library to the place CMAKE usually looks for them
install(TARGETS ${PROJECT_NAME})

# Install the ${PROJECT_NAME}Config.cmake configuration file
install(FILES
        cmake/${PROJECT_NAME}Config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}/cmake)

#building the executable targets of this library
add_executable(3dDynamicsSim_main src/main.cpp)
target_link_libraries(3dDynamicsSim_main PRIVATE 3dDynamicsSim)

add_executable(test_matplotlib src/test_matplotlib.cpp)
target_link_libraries(test_matplotlib 3dDynamicsSim)