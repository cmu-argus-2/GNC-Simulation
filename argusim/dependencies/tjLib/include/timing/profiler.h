#ifndef _TJLIB_TIMING_PROFILER_
#define _TJLIB_TIMING_PROFILER_

#include <iostream>
#include <stack>
#include <string>
#include <vector>

#include "utils.h"

using namespace std::chrono;

// preallocating memory so we dont waste time on allocating more timers. rarely
// use more than 10 timers. Is this really necessary? Is the allocation time
// what kills you or is it something else?
#define BEGINNING_NUMBER_OF_TIMERS 3
typedef struct {
    std::string name;
    timestamp start{};
    timestamp stop{};

    [[nodiscard]] double elapsed() const {
        return ::elapsed(start, stop);
    }
} Timer;

class Profiler {
   private:
    std::vector<Timer> timers_;
    std::stack<uint> stop_idx_stack_;
    std::vector<uint> indent_amounts_;

   public:
    Profiler() {
        timers_.reserve(BEGINNING_NUMBER_OF_TIMERS);
        indent_amounts_.reserve(BEGINNING_NUMBER_OF_TIMERS);
    }

    inline void start(std::string name = "") {
        uint indent_amount = stop_idx_stack_.size();
        indent_amounts_.push_back(indent_amount);

        uint next_timer_to_stop = timers_.size();
        stop_idx_stack_.push(next_timer_to_stop);

        // capture time as late as possible so it is closest to the
        // proceeding code at call site
        Timer tp;
        tp.name  = std::move(name);
        tp.start = std::chrono::steady_clock::now();
        timers_.push_back(tp);
    }

    inline void stop() {
        // capture time immediatly
        timestamp stop = std::chrono::steady_clock::now();

        uint timer_to_stop = stop_idx_stack_.top();
        stop_idx_stack_.pop();
        timers_[timer_to_stop].stop = stop;
    }

    void printSummary() {
        uint number_of_timers = timers_.size();
        for (uint i = 0; i < number_of_timers; i++) {
            uint indent_amount = indent_amounts_[i];
            for (int j = 0; j < indent_amount; j++) {
                std::cout << "\t";
            }

            std::cout << "Timer ";
            Timer tp = timers_[i];
            if (tp.name.empty()) {
                std::cout << i << " ";
            } else {
                std::cout << tp.name << " ";
            }
            printElapsed(tp.start, tp.stop);
        }
    }

    Timer getTimer(const std::string& name) {
        Timer retval;
        for (auto timer : timers_) {
            if (timer.name == name) {
                retval = timer;
                break;
            }
        }
        return retval;
    }
};

#endif