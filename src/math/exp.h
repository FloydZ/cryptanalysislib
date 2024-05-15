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
	/// \tparam T
	/// \param x
	/// \param sum
	/// \param n
	/// \param i
	/// \param t
	/// \return
	__device__ __host__
	template<typename T>
	    requires std::is_floating_point_v<T>
	constexpr T exp(T x, T sum, T n, uint64_t i, T t) {
		return feq(sum, sum + t / n) ? sum : cryptanalysislib::math::exp(x, sum + t / n, n * i, i + T{1}, t * x);
	}

	///
	/// \tparam T
	/// \param x
	/// \return
	__device__ __host__
	template<typename T>
	    requires std::is_arithmetic_v<T>
	constexpr T exp(T x) {
		if constexpr (std::is_integral_v<T>) {
			return cryptanalysislib::math::exp<double>(static_cast<double>(x), 1.0, 1.0, 2, static_cast<double>(x));
		} else {
			return cryptanalysislib::math::exp(x, 1.0, 1.0, 2, x);
		}
	}
}
#endif //CRYPTANALYSISLIB_EXP_H
