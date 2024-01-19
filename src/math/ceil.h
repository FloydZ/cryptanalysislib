#ifndef CRYPTANALYSISLIB_CEIL_H
#define CRYPTANALYSISLIB_CEIL_H

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
#endif//CRYPTANALYSISLIB_CEIL_H
