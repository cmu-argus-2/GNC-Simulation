/**
 * @file StringUtils.h
 * @author Tushaar Jain (tushaarj@andrew.cmu.edu)
 * @brief String parsing utilitites  
 * @date 2024-10-09
 * 
 */
#include <cassert>
#include <stdexcept>
#include <string>

/**
 * @brief Extracts the first number in the string
 * 
 * @tparam T numeric type
 * @param str the string
 * @return T the parsed number 
 * @throws std::runtime_error if string doesn't contain a number
 */
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

/**
 * @brief Parse string to determine if it represents true or false
 * 
 * @param str the string
 * @return true if string matches one of: ['true' 'True' '1']
 * @return false if string matches one of: ['false' 'False' '0']
 * @throws std::invalid_argument otherwise
 */
bool parseBool(const std::string& str);

/**
 * @brief Determines if a string ends with a given suffix
 * 
 * @param str the string
 * @param suffix the suffix
 * @return true if the string ends with the suffix
 * @return false otherwise
 */
bool ends_with(const std::string& str, const std::string& suffix);
