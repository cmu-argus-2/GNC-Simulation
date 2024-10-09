#include <array>
#include <memory>
#include <stdexcept>
#include <string>

std::string exec(const char* cmd) {
    // from: https://stackoverflow.com/a/478960
    std::array<char, 128> buffer{};
    std::string result;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    // TODO(tushaar): fix this line of code so it compiles wihtout need for GCC pragma nonsense
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
#pragma GCC diagnostic pop
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}
