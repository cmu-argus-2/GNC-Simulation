#ifndef _POSE_SIMULATOR_IO_
#define _POSE_SIMULATOR_IO_

#include <optional>

#include "math/vector_math.h"

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

std::string getFullFilename(const std::string &directory_path, const std::string &filename_start,
                            const std::string &filename_end);

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

bool copyFile(const std::string &sourceFilePath, const std::string &destinationFolder);

#endif   // _POSE_SIMULATOR_IO_