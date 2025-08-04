#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_FFS_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_FFS_H

#include "helper.h"


namespace cryptanalysislib {
	/// Find the index of the first set bit (Find First Set)
	/// \tparam T Base integer data type
	/// \param data [in]: Input value to find the first set bit in
	/// \return Position of the first set bit (1-indexed, returns 0 if no bits are set)
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
			assert(false);
            return 0;
		}
	}
}

#endif
