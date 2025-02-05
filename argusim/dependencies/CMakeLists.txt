add_subdirectory(tjLib)
add_subdirectory(pybind11)
include_directories(${SPICE_INCLUDE})
add_library(Spice SHARED IMPORTED)
set_target_properties(Spice PROPERTIES
    IMPORTED_LOCATION "Spice/build/libspice.so"
    INTERFACE_INCLUDE_DIRECTORIES "Spice/include/")
add_subdirectory(Spice)
add_subdirectory(yaml-cpp)
