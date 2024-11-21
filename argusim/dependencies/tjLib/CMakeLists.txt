cmake_minimum_required(VERSION 3.10)

include_directories(include/)

add_subdirectory(src)

# ============== library creation ==============
add_library(tjLib STATIC src/misc.cpp)
target_link_libraries(tjLib tjLogLib tjMathLib tjTimeLib tjStringUtilsLib)
target_include_directories(tjLib PUBLIC include)
