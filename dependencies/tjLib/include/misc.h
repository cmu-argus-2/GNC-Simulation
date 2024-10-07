#ifndef _TJLIB_MISC_
#define _TJLIB_MISC_

#include <iostream>
#define print(x) std::cout << (#x) << ":\n" << (x) << "\n\n";

std::string exec(const char* cmd);

#endif