#ifndef CRYPTANALYSISLIB_MATH_EXP_H
#define CRYPTANALYSISLIB_MATH_EXP_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <type_traits>
#include <cstdint>
#include "abs.h"  // needed for `feq`
#include "helper.h"

namespace cryptanalysislib::math {
	/// exp by Taylor series expansion
	/// NOTE: the term x^i/i! is updated incrementally, as x^i and i! overflow
	///		separately (to inf/inf = NaN) long before the series converges.
	///		Negative arguments are computed as 1/exp(-x) to avoid cancellation.
	/// \tparam T
	/// \param x
	/// \return e^x
	__device__ __host__
	template<typename T>
	    requires std::is_arithmetic_v<T>
	constexpr T exp(T x) {
		if constexpr (std::is_integral_v<T>) {
			return (T)cryptanalysislib::math::exp<double>(static_cast<double>(x));
		} else {
			if (x != x) {
				return x; // NaN
			}

			if (x < T{0}) {
				return T{1} / cryptanalysislib::math::exp<T>(-x);
			}

			T sum = T{1}, term = T{1};
			for (uint64_t i = 1; i < 4096; ++i) {
				term *= x / T(i);
				const T next = sum + term;
				if (next == sum) {
					break;
				}
				sum = next;
			}
			return sum;
		}
	}
}
#endif //CRYPTANALYSISLIB_EXP_H
