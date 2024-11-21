#include "logging/MultiFileLogger.h"

void MultiFileLogger::log(const std::string &file_name, double x, const std::string &label) {
    if (files_.count(file_name) == 0) {   // haven't logged to the given file name
        files_[file_name] = std::ofstream(file_name, std::ios::binary);
        if (files_[file_name].is_open()) {   // Write the header
            files_[file_name] << label << std::endl;
        }
    }

    auto &file = files_[file_name];
    if (file.is_open()) {   // Write the data
        file.write(reinterpret_cast<const char *>(&x), sizeof(double));
        // file.flush();
    }
}
void MultiFileLogger::log(const std::string &file_name, double t, double x, const std::string &t_label,
                          const std::string &x_label) {
    if (files_.count(file_name) == 0) {   // haven't logged to the given file name
        files_[file_name] = std::ofstream(file_name, std::ios::binary);
        if (files_[file_name].is_open()) {   // Write the header
            files_[file_name] << t_label << "," << x_label << std::endl;
        }
    }

    auto &file = files_[file_name];
    if (file.is_open()) {   // Write the data
        file.write(reinterpret_cast<const char *>(&t), sizeof(double));
        file.write(reinterpret_cast<const char *>(&x), sizeof(double));
        // file.flush();
    }
}
void MultiFileLogger::log_on_change_and_timer(const std::string &file_name, double t, double x, double period,
                                              const std::string &t_label, const std::string &x_label) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (prev_values_.count(file_name) == 0 or x != prev_values_[file_name] or t - last_log_time_[file_name] >= period) {
#pragma GCC diagnostic pop
        log(file_name, t, x, t_label, x_label);
        prev_values_[file_name]   = x;
        last_log_time_[file_name] = t;
    }
}

void MultiFileLogger::close_log(const std::string &file_name) {
    if (files_.count(file_name) != 0) {   // file with the given name exists
        auto &file = files_[file_name];
        if (file.is_open()) {
            file.close();
        }
    }
}
void MultiFileLogger::close_all_logs() {
    for (auto it = files_.begin(); it != files_.end(); ++it) {
        auto &file = it->second;
        if (file.is_open()) {
            file.close();
        }
    }
}