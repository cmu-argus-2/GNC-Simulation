#include <cassert>
#include <stdexcept>
#include <string>

template <typename T>
T parseNumber(const std::string& str) {
    assert(not str.empty());

    std::istringstream iss(str);
    T num;

    auto N = str.length();
    // Check each character in the input stream
    for (int i = 0; i < N; i++) {
        char c{};
        iss >> c;
        if ((c >= '0' and c <= '9') or c == '-' or c == '.') {
            if (c == '-') {
                // continue if there is no proceeding character or if it there
                // is, but it's not numeric
                if (i + 1 >= N) {   // there is no proceeding character
                    continue;
                }
                char next_char = str[i + 1];
                if (next_char < '0' || next_char > '9') {
                    continue;
                }
            }
            iss.putback(c);
            if (iss >> num) {
                return num;   // Return the number if successfully parsed
            }
        }
    }

    // If no float was found or correctly parsed
    throw std::runtime_error("No numeric value found in the string");
}

bool parseBool(const std::string& str);

bool ends_with(const std::string& str, const std::string& suffix);
