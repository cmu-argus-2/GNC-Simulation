#include "IO.h"

#include <dirent.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <tuple>

#include "colored_output.h"

vectord SplitCSV(const std::string &data, char separator, char delimiter) {
    vectord Values;
    std::string Val;
    bool VDel     = false;   // Is within delimiter?
    size_t CDel   = 0;       // Delimiters counter within delimiters.
    const char *C = data.c_str();
    size_t P      = 0;
    do {
        if ((Val.length() == 0) && (C[P] == delimiter)) {
            VDel = !VDel;
            CDel = 0;
            P++;
            continue;
        }
        if (VDel) {
            if (C[P] == delimiter) {
                if (((CDel % 2) == 0) &&
                    ((C[P + 1] == separator) || (C[P + 1] == 0) || (C[P + 1] == '\n') || (C[P + 1] == '\r'))) {
                    VDel = false;
                    CDel = 0;
                    P++;
                    continue;
                } else {
                    CDel++;
                }
            }
        } else {
            if (C[P] == separator) {
                Values.push_back(strtod(Val.c_str(), nullptr));
                Val = "";
                P++;
                continue;
            }
            if ((C[P] == 0) || (C[P] == '\n') || (C[P] == '\r')) {
                break;
            }
        }
        Val += C[P];
        P++;
    } while (P < data.length());
    if (!Val.empty()) {
        Values.push_back(strtod(Val.c_str(), nullptr));
    }
    return Values;
}

bool ReadCsv(const std::string &fname, std::vector<vectord> &data, bool skipHeader, char separator, char delimiter) {
    bool Ret = false;
    std::ifstream FCsv(fname);
    assert(FCsv.good());   // make sure the file exists
    if (FCsv) {
        FCsv.seekg(0, std::ifstream::end);
        size_t Siz = FCsv.tellg();
        if (Siz > 0) {
            FCsv.seekg(0);
            data.clear();
            std::string Line;

            if (skipHeader) {   // dummy read
                getline(FCsv, Line, '\n');
            }
            while (getline(FCsv, Line, '\n')) {
                data.push_back(SplitCSV(Line, separator, delimiter));
            }
            Ret = true;
        }
        FCsv.close();
    }
    return Ret;
}

std::string getFullFilename(const std::string &directory_path, const std::string &filename_start,
                            const std::string &filename_end) {
    DIR *dir = opendir(directory_path.c_str());
    if (dir == nullptr) {
        std::cout << ERROR_INFO("Error opening directory: ") << directory_path << std::endl;
        exit(0);
        return "";
    }

    // The pattern for the file you are looking for
    std::string filename_pattern = filename_start + ".*" + filename_end;
    std::cout << "Looking for " << filename_pattern << std::endl;

    std::regex pattern(filename_pattern);

    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);
        // if (filename.find(filename_pattern) != std::string::npos) {
        if (std::regex_match(filename, pattern)) {
            closedir(dir);
            std::string fullFilePath = directory_path + "/" + filename;
            std::cout << SUCCESS_INFO("Found ") << fullFilePath << std::endl;
            return fullFilePath;
        }
    }

    closedir(dir);

    std::cout << ERROR_INFO("Couldn't find pattern \'" << filename_pattern << "\'in directory \'" << directory_path
                                                       << "\'")
              << std::endl;
    exit(0);
    return "";
}

// written by ChatGPT
bool copyFile(const std::string &sourceFilePath, const std::string &destinationFolder) {
    // Find last occurrence of directory separator to extract filename
    size_t pos = sourceFilePath.find_last_of("/\\");
    if (pos == std::string::npos) {
        std::cerr << "Invalid source file path: " << sourceFilePath << std::endl;
        return false;
    }

    // Extract filename from source file path
    std::string filename = sourceFilePath.substr(pos + 1);

    // Construct destination file path by appending filename to destination
    // folder
    std::string destinationPath = destinationFolder + "/" + filename;

    // Open source file for reading
    std::ifstream sourceStream(sourceFilePath, std::ios::binary);
    if (!sourceStream) {
        std::cerr << "Failed to open source file: " << sourceFilePath << std::endl;
        return false;
    }

    // Open destination file for writing
    std::ofstream destinationStream(destinationPath, std::ios::binary);
    if (!destinationStream) {
        std::cerr << "Failed to open destination file: " << destinationPath << std::endl;
        return false;
    }

    // Copy content from source to destination
    destinationStream << sourceStream.rdbuf();

    // Check if copy operation was successful
    if (!destinationStream) {
        std::cerr << "Failed to write to destination file: " << destinationPath << std::endl;
        return false;
    }

    return true;
}

// int main(int argc, char *argv[]) {
//     if (argc != 2) {
//         std::cout << "Expected 1 argument ( path to csv file)" << std::endl;
//         exit(0);
//     }
//     std::vector<vectord> Data;
//     ReadCsv(argv[1], Data);

//     for (const auto &vector : Data) {
//         for (const auto &entry : vector) {
//             std::cout << entry << "|";
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }