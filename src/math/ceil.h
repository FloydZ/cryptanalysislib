#ifndef CRYPTANALYSISLIB_MATH_CEIL_H
#define CRYPTANALYSISLIB_MATH_CEIL_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math.h>`"
#endif

#include <cstdint>
#include "helper.h"


namespace cryptanalysislib::math {

	/// \param num
	/// \return
	__device__ __host__
	constexpr int32_t cceil(float num) {
		return (static_cast<float>(static_cast<int32_t>(num)) == num)
		               ? static_cast<int32_t>(num)
		               : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
	}

	/// \param num
	/// \return
	__device__ __host__
	constexpr int64_t cceil(double num) {
		return (static_cast<double>(static_cast<int32_t>(num)) == num)
		               ? static_cast<int64_t>(num)
		               : static_cast<int64_t>(num) + ((num > 0) ? 1 : 0);
	}
}
#endif//CRYPTANALYSISLIB_CEIL_H
