#ifndef CRYPTANALYSISLIB_TRANSPOSE_EEA_H
#define CRYPTANALYSISLIB_TRANSPOSE_EEA_H

#include <type_traits>
#include <algorithm>

namespace cryptanalysislib {

    /// \param x[out]: 
    /// \param y[out]:
    /// \param a[in]:
    /// \param b[in]:
    /// \return gcd(a, b) and x, y s.t. x*a + y*b = gcd
    template<typename T>
    #if __cplusplus > 201709L
        requires std::is_arithmetic_v<T>
    #endif
    constexpr static T eea(T &x, T &y, const T a, const T b) noexcept {
        T xx = y = 0, yy = x = 1;
        while (b) {
            x -= a / b * xx; std::swap(x, xx);
            y -= a / b * yy; std::swap(y, yy);
            a %= b; std::swap(a, b);
        }
        return a;
    }

} // end namespace cryptanalysislib
#endif
