#ifndef CRYPTANALYSISLIB_MATH_ENTROPY_H
#define CRYPTANALYSISLIB_MATH_ENTROPY_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include "math/log.h"

namespace cryptanalysislib::math {
	/// Entropy function
	/// \param x input
	/// \return H[x] := -x*Log2[x] - (1 - x)*Log2[1 - x];

	__device__ __host__
	template<typename T = double>
		requires std::is_floating_point_v<T>
	constexpr T HH(const T x) noexcept {
		if (x <= 0)
			return 0.;

		if (x >= 1.)
			return 0.;

		return -x * log2(x) - (1. - x) * log2(1. - x);
	}
}
#endif//CRYPTANALYSISLIB_MATH_ENTROPY_H
