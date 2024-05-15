#ifndef CRYPTANALYSISLIB_MATH_ROUND_H
#define CRYPTANALYSISLIB_MATH_ROUND_H

#include <cstdint>
#include "helper.h"

namespace cryptanalysislib::math {
	/// \param num
	/// \return
	__device__ __host__ template<typename T>
	    requires std::is_floating_point_v<T>
	constexpr int64_t round(const T val) {
		return (int64_t) val;
	}
}



#endif //CRYPTANALYSISLIB_ROUND_H
