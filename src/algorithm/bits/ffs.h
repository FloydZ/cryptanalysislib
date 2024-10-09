#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_FFS_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_FFS_H

#include "helper.h"

// TODO move popcount here
namespace cryptanalysislib {
	/// \tparam T base data type
	/// \param data input data type
	/// \return hamming weight (popcount) of the input vector
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_integral<T>::value
#endif
	constexpr inline uint32_t ffs(const T data) noexcept {
		if constexpr(sizeof(T) < 8) {
			return __builtin_ffsl(data);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_ffsll(data);
		} else if constexpr(sizeof(T) == 16) {
			const auto t = __builtin_ffsll(data);
			if (!t) {
				return __builtin_ffsll(data >> 64);
			}

			return t;
		} else {
			ASSERT(false);
		}
	}
}

#endif
