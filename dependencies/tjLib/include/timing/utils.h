#ifndef _TJLIB_TIMING_UTILS_
#define _TJLIB_TIMING_UTILS_

#include <chrono>

typedef std::chrono::time_point<std::chrono::steady_clock> timestamp;

/**
 * @brief returns number of seconds from start to end
 */
double elapsed(timestamp start, timestamp end);

double elapsed_ms(timestamp start, timestamp end);

double elapsed_us(timestamp start, timestamp end);

double elapsed_ns(timestamp start, timestamp end);

void printElapsed(timestamp start, timestamp end, uint8_t decimal_digits = 3);

#endif