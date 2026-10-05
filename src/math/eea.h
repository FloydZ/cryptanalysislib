#ifndef CRYPTANALYSISLIB_TRANSPOSE_EEA_H
#define CRYPTANALYSISLIB_TRANSPOSE_EEA_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

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
    constexpr static T eea(T &x, T &y,
	                       const T a, const T b) noexcept {
        T xx = y = 0, yy = x = 1;
		T aa = a, bb = b;
        while (bb) {
            x -= aa / bb * xx; std::swap(x, xx);
            y -= aa / bb * yy; std::swap(y, yy);
            aa %= bb;
			std::swap(aa, bb);
        }
        return aa;
    }

} // end namespace cryptanalysislib
#endif
