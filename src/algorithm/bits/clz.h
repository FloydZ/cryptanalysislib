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
	/// NOTE: counts relative to the width of `T` (clz<uint32_t>(1) == 31),
	///		and clz(0) is the width of `T`
	constexpr static inline uint32_t clz(const T data) noexcept {
		constexpr uint32_t bits = sizeof(T) * 8u;
		if (data == 0) {
			return bits;
		}

		if constexpr(sizeof(T) <= 4) {
			using U = std::make_unsigned_t<T>;
			return __builtin_clz((uint32_t)(U)data) - (32u - bits);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_clzll((uint64_t)data);
		} else if constexpr(sizeof(T) == 16) {
			const unsigned __int128 d = (unsigned __int128)data;
			const uint64_t hi = (uint64_t)(d >> 64u);
			if (hi != 0) {
				return __builtin_clzll(hi);
			}
			return 64u + __builtin_clzll((uint64_t)d);
		} else {
			assert(false);
			return 0;
		}
	}
}
#endif
