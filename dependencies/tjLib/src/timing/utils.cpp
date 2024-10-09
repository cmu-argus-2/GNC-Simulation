#include "timing/utils.h"

#include <sys/types.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <ratio>

using namespace std::chrono;

double elapsed(timestamp start, timestamp end) {
    return duration<double>(end - start).count();
}

double elapsed_ms(timestamp start, timestamp end) {
    return duration_cast<duration<double, std::milli>>(end - start).count();
}

double elapsed_us(timestamp start, timestamp end) {
    return duration_cast<duration<double, std::micro>>(end - start).count();
}

double elapsed_ns(timestamp start, timestamp end) {
    return duration_cast<duration<double, std::nano>>(end - start).count();
}

void printElapsed(timestamp start, timestamp end, uint8_t decimal_digits) {
    std::cout << "Elapsed time: ";

    double ns         = elapsed_ns(start, end);
    double us         = elapsed_us(start, end);
    double ms         = elapsed_ms(start, end);
    double s          = elapsed(start, end);
    double multiplier = pow(10, decimal_digits);
    if (1 <= s) {
        std::cout << ((uint64_t)(multiplier * s)) / multiplier << "s\n";
    } else if (1 <= ms) {
        std::cout << ((uint64_t)(multiplier * ms)) / multiplier << "ms\n";
    } else if (1 <= us) {
        std::cout << ((uint64_t)(multiplier * us)) / multiplier << "us\n";
    } else {
        std::cout << ((uint64_t)(multiplier * ns)) / multiplier << "ns\n";
    }
}
