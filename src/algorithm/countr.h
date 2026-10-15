#ifndef CRYPTANALYSISLIB_ALGORITHM_COUNTR_H
#define CRYPTANALYSISLIB_ALGORITHM_COUNTR_H

#include <cstdint>
#include <type_traits>

namespace cryptanalysislib {

	/// Counts the number of consecutive 0 bits, starting from the least
	/// significant bit (same semantics as `std::countr_zero`).
	/// NOTE: well defined for `x == 0` (returns the bit width of `T`), unlike
	/// 	`__builtin_ctz`. Needs no `-mbmi`: compilers emit `tzcnt` if the
	/// 	target has it, else `bsf` + a zero check.
	/// \tparam T unsigned integer type (up to 128 bits)
	/// \param x [in]: value to count the trailing zeros of
	/// \return [out]: number of trailing zero bits, `8*sizeof(T)` if `x == 0`
	template<typename T>
#if __cplusplus > 201709L
		requires (std::is_unsigned_v<T> || std::is_same_v<T, unsigned __int128>)
#endif
	[[nodiscard]] constexpr inline uint32_t countr_zero(const T x) noexcept {
		constexpr uint32_t bits = sizeof(T) * 8u;
		if (x == 0) { return bits; }

		if constexpr (sizeof(T) <= sizeof(unsigned int)) {
			return (uint32_t)__builtin_ctz((unsigned int)x);
		} else if constexpr (sizeof(T) <= sizeof(unsigned long long)) {
			return (uint32_t)__builtin_ctzll((unsigned long long)x);
		} else {
			static_assert(sizeof(T) == 16);
			const uint64_t lo = (uint64_t)x;
			if (lo) { return (uint32_t)__builtin_ctzll(lo); }
			return 64u + (uint32_t)__builtin_ctzll((uint64_t)(x >> 64u));
		}
	}
} // namespace cryptanalysislib

#endif
