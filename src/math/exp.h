#ifndef CRYPTANALYSISLIB_EXP_H
#define CRYPTANALYSISLIB_EXP_H

#include <type_traits>
#include <cstdint>

// exp by Taylor series expansion
template <typename T>
    requires std::is_floating_point_v<T>
constexpr T exp(T x, T sum, T n, uint64_t i, T t) {
    return feq(sum, sum + t/n) ?
           sum :
           exp(x, sum + t/n, n * i, i+T{1}, t * x);
}

template<typename T>
    requires std::is_arithmetic_v<T>
constexpr T exp(T x) {
    if constexpr(std::is_integral_v<T>) {
        return exp<double>(static_cast<double>(x), 1.0, 1.0, 2, static_cast<double>(x));
    } else {
        return exp(x, 1.0, 1.0, 2, x);
    }
}

#endif //CRYPTANALYSISLIB_EXP_H
