#ifndef _TJLIB_MULTI_FILE_LOGGER_
#define _TJLIB_MULTI_FILE_LOGGER_
#include <fstream>
#include <unordered_map>

#include "math/EigenWrapper.h"
class MultiFileLogger {
   public:
    void log(const std::string &file_name, double x, const std::string &label);
    void log(const std::string &file_name, double t, double x, const std::string &t_label = "t",
             const std::string &x_label = "x");
    void log_on_change_and_timer(const std::string &file_name, double t, double x, double period,
                                 const std::string &t_label = "t", const std::string &x_label = "x");

    template <size_t N>
    void log(const std::string &file_name, double t, const std::array<double, N> &x, const std::string &t_label,
             const std::array<std::string, N> &x_labels) {
        if (files_.count(file_name) == 0) {   // haven't logged to the given file name
            files_[file_name] = std::ofstream(file_name, std::ios::binary);
            auto &file        = files_[file_name];
            if (file.is_open()) {   // Write the header
                file << t_label;
                for (auto x_label : x_labels) {
                    file << "," << x_label;
                }
                file << std::endl;
            }
        }

        auto &file = files_[file_name];
        if (file.is_open()) {   // Write the data
            file.write(reinterpret_cast<const char *>(&t), sizeof(double));
            file.write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(double));
            // file.flush();
        }
    }

    template <std::size_t N>
    void log(const std::string &file_name, double t, const Eigen::Matrix<double, (int)N, 1> &x,
             const std::string &t_label, const std::array<std::string, N> &x_labels) {
        if (files_.count(file_name) == 0) {   // haven't logged to the given file name
            files_[file_name] = std::ofstream(file_name, std::ios::binary);
            auto &file        = files_[file_name];
            if (file.is_open()) {   // Write the header
                file << t_label;
                for (auto x_label : x_labels) {
                    file << "," << x_label;
                }
                file << std::endl;
            }
        }

        auto &file = files_[file_name];
        if (file.is_open()) {   // Write the data
            file.write(reinterpret_cast<const char *>(&t), sizeof(double));
            file.write(reinterpret_cast<const char *>(x.data()),   // TODO(tushaar): does this work with Eigen?
                       N * sizeof(double));
            // file.flush();
        }
    }

    /*
      void log(const std::string &file_name, double t, Vector3 v,
               const std::string &t_label = "t",
               const std::array<std::string, 3> &labels = {"x", "y", "z"});
      void log(const std::string &file_name, double t, Quaternion q,
               const std::string &t_label = "t",
               const std::array<std::string, 4> &labels = {"w", "x", "y", "z"});
    */

    void close_log(const std::string &file_name);
    void close_all_logs();

   private:
    std::unordered_map<std::string, std::ofstream> files_;
    std::unordered_map<std::string, double> prev_values_;
    std::unordered_map<std::string, double> last_log_time_;
};
#endif