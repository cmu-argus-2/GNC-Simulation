#include <sstream>
#include <stdexcept>
#include <string>

template <typename T>
T parseNumber(const std::string& str) {
    std::istringstream iss(str);
    T num;
    char ch;

    // Check each character in the input stream
    while (iss >> ch) {
        // If the character might be part of a number, put it back and try to
        // read the number
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '.') {
            iss.putback(ch);
            if (iss >> num) {
                return num;   // Return the number if successfully parsed
            }
        }
    }

    // If no float was found or correctly parsed
    throw std::runtime_error("No numeric value found in the string");
}

bool parseBool(const std::string& str) {
    if (str == "true" || str == "True" || str == "1") {
        return true;
    } else if (str == "false" || str == "False" || str == "0") {
        return false;
    } else {
        throw std::invalid_argument("Invalid boolean string: " + str);
    }
}

bool ends_with(const std::string& str, const std::string& suffix) {
    // from https://stackoverflow.com/a/42844629/19497599
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}