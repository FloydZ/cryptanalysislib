#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_CLZ_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_CLZ_H

#include <cstdint>
#include <type_traits>

#include "helper.h"

/// namespace containing popcount algorithms
namespace cryptanalysislib::algorithm {
	/// \tparam T base data type
	/// \param data input data type
	/// \return
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_integral<T>::value
#endif
	constexpr static inline uint32_t clz(const T data) noexcept {
		if constexpr(sizeof(T) < 8) {
			return __builtin_clzl(data);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_clzll(data);
		} else if constexpr(sizeof(T) == 16) {
			if((uint64_t)data == 0) {
				return clz<uint64_t>(data >> 64);
			}
			return clz<uint64_t>(data);
		} else {
			ASSERT(false);
		}
	}
}
#endif
