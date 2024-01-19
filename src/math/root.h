#ifndef CRYPTANALYSISLIB_ROOT_H
#define CRYPTANALYSISLIB_ROOT_H

#include <type_traits>
#include "math/abs.h"

// square root by Newton-Raphson method
template <typename T>
requires std::is_arithmetic_v<T>
constexpr T sqrt(T x, T guess) {
    return  feq(guess, (guess + x/guess) / T{2}) ?
            guess :
            sqrt(x, (guess + x/guess) / T{2});
}

// square root by Newton-Raphson method
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T sqrt(T x) {
    if constexpr(std::is_integral_v<T>)
        return sqrt<double>(x, x);

    return sqrt(x, x);
}

// cube root by Newton-Raphson method
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T cbrt(T x, T guess) {
    return  feq(guess, (T{2}*guess + x/(guess*guess))/T{3}) ?
            guess :
            cbrt(x, (T{2}*guess + x/(guess*guess))/T{3});
}

// cube root by Newton-Raphson method
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr T cbrt(T x) {
    if constexpr(std::is_integral_v<T>)
        return cbrt<double>(x, x);
    return cbrt(x, x);
}

// k root by Newton-Raphson method
template <typename T, typename T2>
    requires std::is_arithmetic_v<T> &&
             std::is_integral_v<T2>
constexpr T kthrt(T x, T guess, const T2 k) {
    return  feq(guess, (static_cast<T>(k-1) * guess + x / ipow(x, k-1) / T{3})) ?
            guess :
            kthrt(x, (static_cast<T>(k-1) * guess + x / ipow(x, k-1)) / static_cast<T>(k), k);
}

// cube root by Newton-Raphson method
template <typename T, typename T2>
    requires std::is_arithmetic_v<T> &&
             std::is_integral_v<T2>
constexpr T kthrt(T x, const T2 k) {
    if constexpr(std::is_integral_v<T>)
        return kthrt<double>(x, x, k);
    return kthrt(x, x, k);
}
#endif //CRYPTANALYSISLIB_ROOT_H
