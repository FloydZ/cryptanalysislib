#pragma once

#include <iostream>
#include <limits>
#include <ranges>
#include <string_view>
#include <type_traits>


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


/// compile time configuration of a `logger`.
/// NOTE: with the defaults everything is printed to `std::cout`, without a
///     prefix, and flushed after every message.
/// NOTE: execution/threading policies (async logging, locking) are not
///     implemented; a message is a single `operator<<` chain on `std::cout`.
struct LoggerConfig {
    /// messages with a lower level are dropped
    const LoggingLevel min_level = LoggingLevel::debug;

    /// flush policy: `true` ends a message with `std::endl`, `false` with '\n'
    const bool flush = true;

    /// prefix every message with its level, e.g. "[warning] "
    const bool print_level = false;
};

inline constexpr LoggerConfig __loggerConfig{};

/// \return the name of a level
[[nodiscard]] constexpr inline const char *to_string(const LoggingLevel l) noexcept {
    switch (l) {
        case LoggingLevel::debug:   return "debug";
        case LoggingLevel::info:    return "info";
        case LoggingLevel::warning: return "warning";
        case LoggingLevel::error:   return "error";
    }
    return "unknown";
}

template<const LoggerConfig &config = __loggerConfig>
class logger {
public:
    template<typename ...T>
    void log(const LoggingLevel &l, T... t){
        if (l < config.min_level) {
            return;
        }

        if constexpr (config.print_level) {
            std::cout << "[" << to_string(l) << "] ";
        }

        ((std::cout << std::forward<T>(t)), ...);
        if constexpr (config.flush) {
            std::cout << std::endl;
        } else {
            std::cout << '\n';
        }
    }

    template<typename ...T>
    void debug(T... t) { log(LoggingLevel::debug, t...); }

    template<typename ...T>
    void info(T... t) { log(LoggingLevel::info, t...); }

    template<typename ...T>
    void warning(T... t) { log(LoggingLevel::warning, t...); }

    template<typename ...T>
    void error(T... t) { log(LoggingLevel::error, t...); }


    logger& operator()(const LoggingLevel &l) noexcept {
        level = l;
        return *this;
    }
    LoggingLevel level = LoggingLevel::debug;
};
}; // end namespace cryptanalysislib::logging


namespace cryptanalysislib {
    ///
    inline auto log = logging::logger();
};

// NOTE: the operators live in the namespace of `logger`/`LoggingLevel`, so
// they are found by ADL without `using namespace cryptanalysislib;`
namespace cryptanalysislib::logging {
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
        cryptanalysislib::log(l) << s;
        return l;
    }
};

namespace cryptanalysislib::logging {
    /// a value together with its name (the key in the json record), see `LOG_DATA(x)`
    template<typename T>
    struct named_data {
        const char *name;
        const T &value;
    };

    /// \return `value` with the key `name`, e.g. `data_logger << named("n", n)`
    template<typename T>
    constexpr named_data<T> named(const char *name, const T &value) noexcept {
        return named_data<T>{name, value};
    }

    /// ends a `DataLogger` record, see `cryptanalysislib::endl`
    struct endl_t {};

    /// writes one json object per record, e.g.
    ///     `data_logger << LOG_DATA(data1) << LOG_DATA(data2) << cryptanalysislib::endl;`
    /// prints `{"data1": <data1>, "data2": <data2>}` and a newline.
    /// Values are written as json: numbers, `true`/`false`, strings (escaped),
    /// ranges as arrays (recursively), everything else via its `operator<<`.
    /// NOTE: C++ cannot see the name of a variable passed to `operator<<`, so
    ///     every value needs a key: `LOG_DATA(x)` (key = "x") or `named("key", x)`.
    /// NOTE: not thread safe
    class DataLogger {
        std::ostream *out;
        bool first = true;

        static void write_string(std::ostream &os, const std::string_view s) {
            constexpr const char *hex = "0123456789abcdef";
            os << '"';
            for (const char c : s) {
                switch (c) {
                    case '"':  os << "\\\""; break;
                    case '\\': os << "\\\\"; break;
                    case '\n': os << "\\n"; break;
                    case '\r': os << "\\r"; break;
                    case '\t': os << "\\t"; break;
                    default:
                        if ((unsigned char)c < 0x20u) {
                            os << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                        } else {
                            os << c;
                        }
                }
            }
            os << '"';
        }

        template<typename T>
        static void write_value(std::ostream &os, const T &v) {
            if constexpr (std::is_same_v<T, bool>) {
                os << (v ? "true" : "false");
            } else if constexpr (std::is_same_v<T, char>) {
                write_string(os, std::string_view(&v, 1));
            } else if constexpr (std::is_integral_v<T>) {
                // `+` prints (u)int8_t as a number
                os << +v;
            } else if constexpr (std::is_floating_point_v<T>) {
                const auto p = os.precision(std::numeric_limits<T>::max_digits10);
                os << v;
                os.precision(p);
            } else if constexpr (std::is_convertible_v<const T &, std::string_view>) {
                write_string(os, std::string_view(v));
            } else if constexpr (std::ranges::range<const T>) {
                os << '[';
                bool f = true;
                for (const auto &e : v) {
                    if (!f) { os << ", "; }
                    f = false;
                    write_value(os, e);
                }
                os << ']';
            } else {
                os << v;
            }
        }

    public:
        DataLogger(std::ostream &os = std::cout) noexcept : out(&os) {}

        template<typename T>
        DataLogger &operator<<(const named_data<T> &d) {
            *out << (first ? "{" : ", ");
            first = false;
            write_string(*out, d.name);
            *out << ": ";
            write_value(*out, d.value);
            return *this;
        }

        /// closes the current record
        DataLogger &operator<<(const endl_t) {
            if (first) { *out << '{'; }
            *out << '}' << std::endl;
            first = true;
            return *this;
        }
    };
}; // end namespace cryptanalysislib::logging

namespace cryptanalysislib {
    /// see `logging::DataLogger`
    inline auto data_logger = logging::DataLogger();

    /// ends a `data_logger` record
    inline constexpr logging::endl_t endl{};
};

/// value with its expression as key: `data_logger << LOG_DATA(data1)`
/// writes `"data1": <data1>`
#define LOG_DATA(x) ::cryptanalysislib::logging::named(#x, (x))
