#include <iostream>
#include <random>

#include "IO.h"
#include "ParameterParser.h"
#include "simulate_trial.h"
#include "tjLib/include/colored_output.h"

std::string trial_directory;

int main() {
    auto parameter_filepath = get_env_var<std::string>("PARAMETER_FILEPATH");
    if (not parameter_filepath.has_value()) {
        std::cerr << ERROR_INFO(
                         "Error: Didn't receive PARAMETER_FILEPATH "
                         "environment variable")
                  << std::endl;
        exit(1);
    }

    Simulation_Parameters params;
    params.getParamsFromFileAndSample(parameter_filepath.value());

    auto LOGGING_DIR = get_env_var<std::string>("TRIAL_DIRECTORY");

    bool exit_early = false;
    if (not LOGGING_DIR.has_value()) {
        std::cerr << ERROR_INFO(
                         "Error: Didn't receive TRIAL_DIRECTORY "
                         "environment variable")
                  << std::endl;
        exit_early = true;
    }
    if (exit_early) {
        exit(3);
    }

    trial_directory = LOGGING_DIR.value();

    params.dumpSampledParametersToYAML(trial_directory + "/sampled_params.yaml");

    simulate_trial(params);

    return 0;
}