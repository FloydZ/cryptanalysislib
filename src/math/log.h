#ifndef CRYPTANALYSISLIB_LOG_H
#define CRYPTANALYSISLIB_LOG_H

#include <type_traits>

#include "math/abs.h"
#include "math/exp.h"

template<typename T>
    requires std::is_floating_point_v<T>
constexpr T log_iter(T x, T y) {
    return y + T{2} * (x - exp(y)) / (x + exp(y));
}

template <typename T>
    requires std::is_floating_point_v<T>
constexpr T log(T x, T y) {
    return  feq(y, log_iter(x, y)) ?
            y :
            log(x, log_iter(x, y));
}

constexpr long double e(){
    return 2.71828182845904523536l;
}

// For numerical stability, constrain the domain to be x > 0.25 && x < 1024
// - multiply/divide as necessary. To achieve the desired recursion depth
// constraint, we need to account for the max double. So we'll divide by
// e^5. If you want to compute a compile-time log of huge or tiny long
// doubles, YMMV.

// if x <= 1, we will multiply by e^5 repeatedly until x > 1
template <typename T>
    requires std::is_floating_point_v<T>
constexpr T logGT(T x) {
    return  x > T{0.25} ?
            log(x, T{0}) :
            logGT<T>(x * e() * e() * e() * e() * e()) - T{5};
}

// if x >= 2e10, we will divide by e^5 repeatedly until x < 2e10
template <typename T>
    requires std::is_floating_point_v<T>
constexpr T logLT(T x) {
    return  x < T{1024} ?
            log(x, T{0}) :
            logLT<T>(x / (e() * e() * e() * e() * e())) + T{5};
}

template<typename T>
    requires std::is_arithmetic_v<T>
constexpr T log(T x) {
    if (x > T{1024}) {
        if constexpr(std::is_integral_v<T>) {
            return logLT<double>(x);
        } else {
            return logLT<double>(x);
        }
    } else {
        if constexpr(std::is_integral_v<T>) {
            return logGT<double>(x);
        } else {
            return logGT<double>(x);
        }
    }
}

#endif //CRYPTANALYSISLIB_LOG_H
