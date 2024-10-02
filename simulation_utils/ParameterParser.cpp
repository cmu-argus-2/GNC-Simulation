#include "ParameterParser.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "ExpressionEvaluation.h"
#include "StringUtils/StringUtils.h"
#include "colored_output.h"
#include "math/conversions.h"
#include "math/random.h"
#include "misc.h"
#include "utils_and_transforms.h"

// FSW includes

// ==========================================================================
// ============================ Helper Functions ============================
// ==========================================================================
double UTC_to_unix(const std::string& UTC_date) {
    std::string command             = "date --utc --date=\'" + UTC_date + "\' +%s";
    std::string output              = exec(command.c_str());
    double seconds_since_unix_epoch = std::stod(output);
    return seconds_since_unix_epoch;
}
// ==========================================================================

void Simulation_Parameters::getParamsFromFileAndSample(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::string error_message = "Failed to open file: " + filename;
        throw std::runtime_error(error_message);
    }
    printf(SUCCESS_INFO("Parsing Parameter file: %s\n"), filename.c_str());

    std::map<std::string, std::string> parameterMap;

    std::string line;
    while (std::getline(file, line)) {
        // parsing each line as "<NAME>: <VALUE>"
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string parameter_name   = line.substr(0, pos);
            std::string parameter_value  = line.substr(pos + 1);
            parameterMap[parameter_name] = parameter_value;
        }
    }
    const std::vector<std::string> expected_parameters = {
        "MAX_TIME",
        "dt",
        "RAAN",
        "init_altitute",
        "orbital_velocity",
        "orbital_incliniation",
        "satellite_mass",
        //
        "Ixx",
        "Iyy",
        "Izz",
        "Ixy",
        "Ixz",
        "Iyz",
        //
        "earliest_sim_start_time_UTC",
        "latest_sim_start_time_UTC",
        "EARTH_RADIUS",
    };
    bool received_all_expected_parameters = true;
    for (const auto& parameter_name : expected_parameters) {
        auto iterator_to_parameter_value = parameterMap.find(parameter_name);
        if (iterator_to_parameter_value == parameterMap.end()) {
            printf(ERROR_INFO("Expected parameter \"%s\" in parameter file "
                              "but didn't find it\n"),
                   parameter_name.c_str());
            received_all_expected_parameters = false;
            continue;
        }

        // ensure paramter isn't blank or whitespace
        std::string parameter_value = iterator_to_parameter_value->second;

        // Find the first non-whitespace character
        size_t start = parameter_value.find_first_not_of(" \t\n\r");

        // Find the last non-whitespace character
        size_t end = parameter_value.find_last_not_of(" \t\n\r");

        if (start == std::string::npos || end == std::string::npos) {
            // The string contains only whitespace characters, or it's
            // empty.
            std::cout << "Blank value provided for parameter \"" << parameter_name << "\"" << std::endl;
            received_all_expected_parameters = false;
        } else {
            parameterMap[parameter_name] =
                parameter_value.substr(start, end - start + 1);   // Extract the substring without
                                                                  // leading/trailing whitespace
        }
    }
    if (not received_all_expected_parameters) {
        std::string error_message = "Failed to read struct data from " + filename;
        throw std::runtime_error(error_message);
    }

    // clang-format off
    MAX_TIME = evaluate(parameterMap["MAX_TIME"]);
    dt = evaluate(parameterMap["dt"]);
    RAAN = evaluate(parameterMap["RAAN"]);
    init_altitute = evaluate(parameterMap["init_altitute"]);
    orbital_velocity = evaluate(parameterMap["orbital_velocity"]);
    orbital_incliniation = evaluate(parameterMap["orbital_incliniation"]);
    satellite_mass = evaluate(parameterMap["satellite_mass"]);
    //
    double Ixx = evaluate(parameterMap["Ixx"]);
    double Iyy = evaluate(parameterMap["Iyy"]);
    double Izz = evaluate(parameterMap["Izz"]);
    double Ixy = evaluate(parameterMap["Ixy"]);
    double Ixz = evaluate(parameterMap["Ixz"]);
    double Iyz = evaluate(parameterMap["Iyz"]);
    //
    earliest_sim_start_unix = UTC_to_unix(parameterMap["earliest_sim_start_time_UTC"]);
    latest_sim_start_unix = UTC_to_unix(parameterMap["latest_sim_start_time_UTC"]);
    EARTH_RADIUS = evaluate(parameterMap["EARTH_RADIUS"]);

    InertiaTensor <<Ixx,Ixy,Ixz,//
                    Ixy,Iyy,Iyz,//
                    Ixz,Iyz,Izz;

    // Verifying parameters that were read in
    {

        assert(0 < MAX_TIME);
        assert(0 < dt);
        assert(0 <= RAAN);
        assert(0 < init_altitute);
        assert(0 < orbital_velocity);
        assert(0 <= orbital_incliniation);
        assert(0 < satellite_mass);
        assert(0 < earliest_sim_start_unix);
        assert(0 < latest_sim_start_unix);
        assert(0 < EARTH_RADIUS);

        // TODO(tushaar): more rigorous check to ensure it is PSD
        assert(InertiaTensor.determinant() > 0.0);

    }   // clang-format on

    // TODO(tushaar): verify math here
    orbital_plane_normal = Vector3{sin(orbital_incliniation) * sin(RAAN),    //
                                   -sin(orbital_incliniation) * cos(RAAN),   //
                                   cos(orbital_incliniation)};
}

void Simulation_Parameters::dumpSampledParametersToYAML(const std::string& absolute_filename) {
    std::ofstream file(absolute_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file \"" << absolute_filename << "\"for writing." << std::endl;
        return;
    }

    file << "MAX_TIME: " << MAX_TIME << std::endl;
    file << "dt: " << dt << std::endl;
    file << "RAAN: " << RAAN << std::endl;
    file << "init_altitute: " << init_altitute << std::endl;
    file << "orbital_velocity: " << orbital_velocity << std::endl;
    file << "orbital_incliniation: " << orbital_incliniation << std::endl;
    file << "satellite_mass: " << satellite_mass << std::endl;
    file << "earliest_sim_start_unix: " << earliest_sim_start_unix << std::endl;
    file << "latest_sim_start_unix: " << latest_sim_start_unix << std::endl;
    file << "EARTH_RADIUS: " << EARTH_RADIUS << std::endl;

    file.close();
}