#ifndef CRYPTANALYSISLIB_MATH_FLOOR_H
#define CRYPTANALYSISLIB_MATH_FLOOR_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <cstdint>
#include "helper.h"

namespace cryptanalysislib::math {
	/// \param num
	/// \return
	__device__ __host__ template<typename T>
	    requires std::is_floating_point_v<T>
	constexpr T floor(const T val) {
		// casting to int truncates the value, which is floor(val) for positive values,
		// but we have to substract 1 for negative values (unless val is already floored == recasted int val)
		const auto val_int = (int64_t) val;
		const T fval_int = (T) val_int;
		return (val >= (T) 0 ? fval_int : (val == fval_int ? val : fval_int - (T) 1));
	}
}
#endif//CRYPTANALYSISLIB_MATH_FLOOR_H
