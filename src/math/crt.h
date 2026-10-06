#pragma once

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <utility>

#include "math/eea.h"
#include "math/mod.h"


namespace cryptanalysislib {

	/// TODO maybe SIMD version? like 8 crt in parallel?
    /// Chinese Remainder Theorem: returns (u, v) s.t.
    /// x=u (mod v) <=> x=a (mod n) and x=b (mod m)
	/// \tparam T signed integer type
	/// \param a
	/// \param n modulus, 1 <= n <= 1e9
	/// \param b
	/// \param m modulus, 1 <= m <= 1e9
	/// \return (u, v) with 0 <= u < v = lcm(n, m), or (0, -1) if there is no solution
    template<typename T>
        requires std::is_signed_v<T>
    std::pair<T, T> crt(T a, T n,
	                    T b, T m) noexcept {
		// s*n + t*m == d == gcd(n, m)
        T s, t;
        const T d = eea<T>(s, t, n, m);
        if ((a - b) % d) {
            return { 0, -1 };
        }

        // x = a + n*k with k = ((b - a)/d * s) mod (m/d)
        const T md = m / d;
        const T k = mulmod<T>(pmod<T>((b - a) / d, md), pmod<T>(s, md), md);
        const T l = n / d * m;
        return { pmod<T>(a + n * k, l), l };
    }
};
