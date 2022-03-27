#ifndef CRYPTANALYSISLIB_IPOW_H
#define CRYPTANALYSISLIB_IPOW_H

#include <type_traits>

template <typename T, typename T2>
    requires std::is_arithmetic_v<T> &&
             std::is_integral_v<T2>
constexpr T ipow(T x, T2 n) {
    return (n == 0) ? T{1} :
           n == 1 ? x :
           n > 1 ? ((n & 1) ? x * ipow(x, n-1) : ipow(x, n/2) * ipow(x, n/2)) :
           T{1} / ipow(x, -n);
}

#endif //CRYPTANALYSISLIB_IPOW_H
