#ifndef CRYPTANALYSISLIB_SIMD_BITS_H
#define CRYPTANALYSISLIB_SIMD_BITS_H

#include <cstdint>

/// Deposit contiguous low bits from unsigned 64-bit integer a to `dst` at the
/// corresponding bit locations specified by `mask`;
/// all other bits in `dst` are set to zero.
template<typename T>
constexpr inline T pdep(const T a, const T mask) noexcept {
#ifdef USE_AVX2 
	if constexpr (sizeof(T) == 4) {
		return _pdep_u32(a, mask);
	} else if constexpr (sizeof(T) == 8) {
		return _pdep_u64(a, mask);
	}
#endif
	T tmp = a, dst = 0;
	uint32_t m = 0, k = 0;

	while (m < 64) {
		if (mask & (1u << k)) {
			dst ^= tmp & (1u << k);
			k += 1;
		}

		m += 1;
	}

	return dst;
}

// TODO
// using _pdep_u32 = pdep<uint32_t>;
#endif
