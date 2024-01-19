#ifndef CRYPTANALYSISLIB_MATH_ROUND_H
#define CRYPTANALYSISLIB_MATH_ROUND_H

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
constexpr int64_t round(const T val) {
	return (int64_t)val;
}



#endif //CRYPTANALYSISLIB_ROUND_H
