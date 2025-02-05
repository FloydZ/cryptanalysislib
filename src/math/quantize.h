#pragma once 

#include <cstdint>
#include <cmath>  // floor()


// In f[] set each element x to q*floor(1/q*(x+q/2))
// E.g.: q=1 ==> round to nearest integer
//       q=1/1000 ==> round to nearest multiple of 1/1000
// For inexact types (float or double).
template  <typename Type>
constexpr void quantize(const Type *f,
                        const size_t n,
                        const double q) noexcept {
    Type qh = q * 0.5;
    Type q1 = 1.0 / q;
    for (uint32_t i = 0; i < n; i++) {
        f[i] = q * floor(q1 * (f[i]+qh));
    }
}
