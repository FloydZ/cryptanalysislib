#ifndef CRYPTANALYSISLIB_MATH_FLOOR_H
#define CRYPTANALYSISLIB_MATH_FLOOR_H

/// add fake CUDA header
#ifndef __device__
#define  __device__
#endif
#ifndef __host__
#define  __host__
#endif

#include <cstdint>

/// \param num
/// \return
__device__ __host__
template <typename T>
	requires std::is_floating_point_v<T>
constexpr T floor(const T val) {
	// casting to int truncates the value, which is floor(val) for positive values,
	// but we have to substract 1 for negative values (unless val is already floored == recasted int val)
	const auto val_int = (int64_t)val;
	const T fval_int = (T)val_int;
	return (val >= (T)0 ? fval_int : (val == fval_int ? val : fval_int - (T)1));
}
#endif//CRYPTANALYSISLIB_MATH_FLOOR_H
