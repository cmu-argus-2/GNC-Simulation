add_library(sim_utils SHARED 
    ExpressionEvaluation.cpp
    ParameterParser.cpp 
    IO.cpp
)
target_link_libraries(sim_utils PUBLIC Eigen3::Eigen tjLib math yaml-cpp::yaml-cpp actuators models)
target_include_directories(sim_utils PUBLIC .)


pybind11_add_module(pysim_utils SHARED 
    ExpressionEvaluation.cpp
    ParameterParser.cpp 
    IO.cpp
)
target_link_libraries(pysim_utils PUBLIC Eigen3::Eigen tjLib math yaml-cpp::yaml-cpp actuators models)
target_include_directories(pysim_utils PUBLIC .)
target_compile_definitions(pysim_utils PRIVATE USE_PYBIND_TO_COMPILE)
