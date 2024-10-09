/**
 * @file IO.h
 * @author Tushaar Jain (tushaarj@andrew.cmu.edu)
 * @brief Various random utilities to help with IO (getting environment variables, reading csv files, completeing
 * filepath based on start and end hints, copying files, etc)
 * @date 2024-10-09
 *
 */
#ifndef _POSE_SIMULATOR_IO_
#define _POSE_SIMULATOR_IO_

#include <optional>

#include "math/vector_math.h"

/**
 * @brief convert a single row of csv into a vector fo doubles 
 * 
 * @param data the csv row as a string
 * @param separator 
 * @param delimiter 
 * @return vectord std::Vector<double> of parsed doubles
 */
vectord SplitCSV(const std::string &data, char separator, char delimiter);

/**
 * @brief
 *
 * @param [in] fname absolute filepath
 * @param [out] data vector of rows
 * @param [in] skipHeader
 * @param [in] separator
 * @param [in] delimiter
 * @return true if successful
 * @return false otherwise
 */
bool ReadCsv(const std::string &fname, std::vector<vectord> &data, bool skipHeader = false, char separator = ',',
             char delimiter = '\"');

/**
 * @brief Get the Full Filename from start and end hints
 * 
 * @param directory_path path to the fodler in which to search for a file
 * @param filename_start what the file should start with
 * @param filename_end what the file should end with
 * @return std::string the full filepath (including directory path) to the file that was found
 */
std::string getFullFilename(const std::string &directory_path, const std::string &filename_start,
                            const std::string &filename_end);


/**
 * @brief Get the environment variable by name 
 * 
 * @tparam T type of value to be returned (int, char, double, string, etc)
 * @param environmental_variable_name name of the env var
 * @return std::optional<T> the value of the env var if it could be parsed. Else, {}
 */
template <typename T>
std::optional<T> get_env_var(const std::string &environmental_variable_name) {
    const char *value = std::getenv(environmental_variable_name.c_str());
    if (value != nullptr) {
        // Environment variable exists, use its value
        T result;
        std::stringstream ss(value);
        ss >> result;

        if (!ss.fail()) {   // apparently this is not equivalent to ss.good()
            return result;
        }
    }
    return std::optional<T>{};
}

/**
 * @brief Basically a programatic way of doing the `cp' command
 * 
 * @param sourceFilePath 
 * @param destinationFolder 
 * @return true if the operation was successfull
 * @return false otherwise (source file not found, or any other IO error)
 */
bool copyFile(const std::string &sourceFilePath, const std::string &destinationFolder);

#endif   // _POSE_SIMULATOR_IO_