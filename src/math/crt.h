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
	/// \tparam T
	/// \tparam F_EEA
	/// \tparam F_MOD
	/// \param a
	/// \param n
	/// \param b
	/// \param m
	/// \param EEA
	/// \param MOD
	/// \return
    template<typename T,
	         typename F_EEA,
	         typename F_MOD>
    std::pair<T, T> crt(T a, T n,
	                    T b, T m,
	                    F_EEA &EEA=eea<T>,
	                    F_MOD &MOD=mod<T>) noexcept {
		// n,m <= 1e9
        T s, t, d = EEA(n, m, s, t);
        if ((a - b) % d) {
            return { 0, -1 };
        }

        return { mod(s*b%m*n + t*a%n*m, n*m)/d, n*m/d };
    }
};
