add_executable(main main.cpp simulate_trial.cpp)

target_link_libraries(main PRIVATE physics sim_utils)
