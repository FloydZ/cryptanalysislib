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

#ifdef USE_AVX2
/// \param mask
/// \return an avx register containing the i-th bit of the input zero extend to 32bits
// 				in the i-th 32bit limb
inline __m256i bit_mask_64_avx2(const uint64_t mask) noexcept {
	ASSERT(mask < (1u << 8u));

	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	expanded_mask *= 0xFFU;
	// the identity shuffle for vpermps, packed to one index per byte
	const uint64_t identity_indices = 0x0706050403020100;
	uint64_t wanted_indices = identity_indices & expanded_mask;

	// copies the input into the lower 64bits of the sse register
	const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	// Zero extend packed unsigned 8-bit integers in "a" to packed
	// 32-bit integers, and store the results in "dst".
	const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
	return shufmask;
}
#endif

/// \param mask
/// \return an avx register containing the i-th bit of the input zero extend to 32bits
// 				in the i-th 32bit limb
constexpr inline uint32x8_t bit_mask_64(const uint64_t mask) noexcept {
	ASSERT(mask < (1u << 8u));

	uint64_t expanded_mask = pdep<uint64_t>(mask, 0x0101010101010101);
	expanded_mask *= 0xFFU;
	// the identity shuffle for vpermps, packed to one index per byte
	const uint64_t identity_indices = 0x0706050403020100;
	uint64_t wanted_indices = identity_indices & expanded_mask;

	// copies the input into the lower 64bits of the sse register
	cryptanalysislib::_uint64x2_t bytevec;
	bytevec[0] = wanted_indices;

	const cryptanalysislib::_uint8x16_t bytevec2 = bytevec;
	// Zero extend packed unsigned 8-bit integers in "a" to packed
	// 32-bit integers, and store the results in "dst".
	uint32x8_t shufmask = uint32x8_t::cvtepu8(bytevec2);
	return shufmask;
}
// using _pdep_u32 = pdep<uint32_t>;
#endif
