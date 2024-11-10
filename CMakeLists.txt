cmake_minimum_required(VERSION 3.12)
project(GNC_Sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

add_compile_options(
    -O3
    -g
    -pedantic               # Issue all the warnings demanded by strict ISO C
    -Wall                   # Warn about most questionable operations
    -Wextra                 # Even more warnings
    -Wwrite-strings         # Warn if not treating string literals as "const"
    -Wpointer-arith         # Warn about suspicious pointer operations
    -Wshadow                # Warn about variable shadowing
    -Wfloat-equal           # Warn about testing floats with ==
    -Werror
    -Wparentheses
    -Wunused-variable
    -Wunused-function
    -Wreturn-type
    -Wunused-value
    -std=c++2a
    #-Wswitch-default
    -Wuninitialized
    -Winit-self
    -Wno-comment            # Don't warn about /* in comments, it's common in the codebase and seems fine
    # -Wno-sign-compare       # Don't warn about sign compare; too many false positives even in external libraries
    # -Wno-unused-parameter   # Happens more frequently than you would think and it's generally ok
)

add_subdirectory(dependencies)
add_subdirectory(world)
add_subdirectory(sensors)
add_subdirectory(simulation_utils)
add_subdirectory(actuators)