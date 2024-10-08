add_library(models_p1 OBJECT gravity.cpp SRP.cpp drag.cpp MagneticField.cpp)
target_link_libraries(models_p1 OBJECT math Eigen3::Eigen tjLib spice)

# Fortran compile
SET (CMAKE_Fortran_COMPILER  gfortran)
ENABLE_LANGUAGE(Fortran)
add_library(models_p2 OBJECT ../../../data/igrf13.f)
set_source_files_properties(../../../data/igrf13.f PROPERTIES COMPILE_FLAGS -Wno-error)

add_library(models SHARED $<TARGET_OBJECTS:models_p1> $<TARGET_OBJECTS:models_p2>)

target_include_directories(models PUBLIC ".")