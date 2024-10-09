add_library(tjMathLib STATIC math/conversions.cpp math/vector_math.cpp
                              math/random.cpp)
target_compile_options(tjMathLib PUBLIC -fPIC)

add_library(tjLogLib STATIC logging/MultiFileLogger.cpp)
target_link_libraries(tjLogLib tjMathLib)
target_compile_options(tjLogLib PUBLIC -fPIC)

add_library(tjTimeLib STATIC timing/utils.cpp)
target_compile_options(tjTimeLib PUBLIC -fPIC)

add_library(tjStringUtilsLib STATIC StringUtils/StringUtils.cpp)
target_compile_options(tjStringUtilsLib PUBLIC -fPIC)
