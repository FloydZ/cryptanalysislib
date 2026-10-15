#ifndef CRYPTANALYSISLIB_ALGORITHM_COUNTL_H
#define CRYPTANALYSISLIB_ALGORITHM_COUNTL_H

#include <cstdint>
#include <type_traits>

namespace cryptanalysislib {

	/// Counts the number of consecutive 0 bits, starting from the most
	/// significant bit (same semantics as `std::countl_zero`).
	/// NOTE: well defined for `x == 0` (returns the bit width of `T`), unlike
	/// 	`__builtin_clz`. Needs no `-mlzcnt`: compilers emit `lzcnt` if the
	/// 	target has it, else `bsr` + a zero check.
	/// \tparam T unsigned integer type (up to 128 bits)
	/// \param x [in]: value to count the leading zeros of
	/// \return [out]: number of leading zero bits, `8*sizeof(T)` if `x == 0`
	template<typename T>
#if __cplusplus > 201709L
		requires (std::is_unsigned_v<T> || std::is_same_v<T, unsigned __int128>)
#endif
	[[nodiscard]] constexpr inline uint32_t countl_zero(const T x) noexcept {
		constexpr uint32_t bits = sizeof(T) * 8u;
		if (x == 0) { return bits; }

		if constexpr (sizeof(T) <= sizeof(unsigned int)) {
			// the builtin counts in 32 bits: remove the extra leading zeros
			constexpr uint32_t extra = sizeof(unsigned int) * 8u - bits;
			return (uint32_t)__builtin_clz((unsigned int)x) - extra;
		} else if constexpr (sizeof(T) <= sizeof(unsigned long long)) {
			constexpr uint32_t extra = sizeof(unsigned long long) * 8u - bits;
			return (uint32_t)__builtin_clzll((unsigned long long)x) - extra;
		} else {
			static_assert(sizeof(T) == 16);
			const uint64_t hi = (uint64_t)(x >> 64u);
			if (hi) { return (uint32_t)__builtin_clzll(hi); }
			return 64u + (uint32_t)__builtin_clzll((uint64_t)x);
		}
	}
} // namespace cryptanalysislib

#endif
