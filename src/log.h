#pragma once

#include <iostream>


// features I want:
//  - simple to use: 
//  ```C++
//  using namespace cryptanalysislib::logging;
//  log::debug("kel"); // oder
//  log::debug << "kel";
//  log::info(C); // <-- for this we need a serialisation of classes
//  ```
//  - basically everything this has: https://github.com/abumq/easyloggingpp
//  gute base source:
//      https://github.com/bschiffthaler/BSlogger/blob/master/src/BSlogger.hpp

namespace cryptanalysislib::logging {

enum LoggingLevel {
    debug = 0,
    info,
    warning,
    error,
};


class LoggerConfig {
    // TODO should container
    // constexpr static ExecutionPolicy __executionPolicy{};
    // constexpr static FlushPolicy     __flushPolicy{};
    // constexpr static ThreadingPolicy __threadingPolicy{};
};

constexpr static LoggerConfig __loggerConfig{};

template<const LoggerConfig &config = __loggerConfig>
class logger {
public:
    template<typename ...T>
    void log(const LoggingLevel &l, T... t){
        ((std::cout << std::forward<T>(t)), ...);
        std::cout << std::endl;
    }

    template<typename ...T>
    void info(T... t) { log(LoggingLevel::info, t...); }


    logger& operator()(const LoggingLevel &l) noexcept {
        level = l;
        return *this;
    }
    LoggingLevel level = LoggingLevel::debug;
};
}; // end namespace cryptanalysislib::logging


namespace cryptanalysislib {
    ///
    auto log = logging::logger();

    template<const logging::LoggerConfig &config=logging::__loggerConfig>
    logging::logger<config>& operator<<(logging::logger<config>& l,
                                        const logging::LoggingLevel& level) {
        l.level = level;
        return l;
    }
    
    template<typename T, const logging::LoggerConfig &config=logging::__loggerConfig>
    logging::logger<config>& operator<<(logging::logger<config>& l,
                                        const T& s) {
        l.log(l.level, s);
        return l;
    }
    
    template<typename T>
    logging::LoggingLevel operator<<(logging::LoggingLevel l,
                                      const T &s) {
        log(l) << s;
        return l;
    }
};
