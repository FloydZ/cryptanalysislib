#ifndef CRYPTANALYSISLIB_SIMD_SHUFFLE_H
#define CRYPTANALYSISLIB_SIMD_SHUFFLE_H

#ifndef USE_AVX2
#error "no avx"
#endif

#include <immintrin.h>

#include "helper.h"

/// \param mask
/// \return an avx register containing the i-th bit of the input zero extend to 32bits
// 				in the i-th 32bit limb
inline __m256i bit_mask_64(const uint64_t mask) noexcept {
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

/// returns a permutation that shuffles down a mask on 32bit limbs
/// e.g INPUT: 0b10000001
///				<-    256    ->
///    OUTPUT: [  0 ,..., 7, 0]
///		   MSB <-32->
/// 		   <-  8 limbs   ->
/// to apply the resulting __m256i permutation use:
///			const uint64_t shuffle = 0b1000001;
/// 		const __m256i permuted_data = _mm256_permutevar8x32_ps(data, shuffle);
/// \param mask bit mask. Must be smaller than 2**8
/// \return the permutation
inline __m256i shuffle_down_32(const uint64_t mask) noexcept {
	// make sure only sane inputs make it.
	ASSERT(mask < (1u << 8u));

	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	// mask |= mask<<1 | mask<<2 | ... | mask<<7;
	expanded_mask *= 0xFFU;
	// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

	// the identity shuffle for vpermps, packed to one index per byte
	const uint64_t identity_indices = 0x0706050403020100;
	uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

	const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
	return shufmask;
}

/// returns a permutation that shuffles down a mask on 32bit limbs
/// e.g INPUT: 0b1001
///				<-     256   ->
///    OUTPUT: [  0  , 0, 3, 0]
///		   MSB <-64->
/// 		   <-  4 limbs   ->
/// to apply the resulting __m256i permutation use:
///			const uint64_t shuffle = 0b1000001;
/// 		const __m256i permuted_data = _mm256_permutevar4x64_pd(data, shuffle);
/// \param mask bit mask. Must be smaller than 2**4
/// \return the permutation
const __m256i shuffle_down_64(const uint64_t mask) noexcept {
	// make sure only sane inputs make it.
	ASSERT(mask < (1u << 4u));

	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	// mask |= mask<<1 | mask<<2 | ... | mask<<7;
	expanded_mask *= 0xFFU;
	// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

	// the identity shuffle for vpermps, packed to one index per byte
	const uint64_t identity_indices = 0x0706050403020100;
	uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

	const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	const __m256i shufmask = _mm256_cvtepu8_epi64(bytevec);
	return shufmask;
}

/// pretty much the same as `shuffle_down_64` but accepts permutation mask bigger than 2**4 up to
/// 2**8, meaning this function returns 2 permutations for at most 2 * 4  uint64_t limbs.
/// \param higher: output parameterm, contains the higher/last 4 permutations
/// \param lower:  output parameter, contain the lower/first 4 permutations
/// \param mask: input parameter
const void shuffle_down_2_64(__m256i &higher, __m256i &lower, const uint64_t mask) noexcept {
	// make sure only sane inputs make it.
	ASSERT(mask < (1u << 8u));

	/// see the description of this magic in `shuffle_down_64`
	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	expanded_mask *= 0xFFU;
	const uint64_t identity_indices = 0x0302010003020100;
	uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

	const __m128i bytevec1 = _mm_cvtsi32_si128(uint16_t(wanted_indices));
	const __m128i bytevec2 = _mm_cvtsi32_si128(uint16_t(wanted_indices >> 16));
	lower = _mm256_cvtepu8_epi64(bytevec1);
	higher = _mm256_cvtepu8_epi64(bytevec2);
}

/// same as shuffle up, but instead a compressed array is expanded according to mask
/// EXAMPLE: INPUT: 0b10100001
/// 		<-      256        ->
/// OUTPUT: [  2  , 0, 1, ..., 0]
///			<-32->
///			<-    8  limbs     ->
/// USAGE:
///			const uint64_t shuffle = 0b1000001;
/// 		const __m256i permuted_data = _mm256_permutevar8x32_ps(data, shuffle);
/// \param mask
/// \return
const __m256i shuffle_up_32(const uint64_t mask) noexcept {
	ASSERT(mask < (1u << 8u));

	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	expanded_mask *= 0xFFU;
	const uint64_t identity_indices = 0x0706050403020100;
	uint64_t wanted_indices = _pdep_u64(identity_indices, expanded_mask);

	const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
	return shufmask;
}

/// same as shuffle up, but instead a compressed array is expanded according to mask
/// EXAMPLE: INPUT: 0b1010
/// 		<-     256    ->
/// OUTPUT: [  1  , 0, 0, 0]
///			<-64->
///			<-   4 limbs  ->
/// USAGE:
///			const uint64_t shuffle = 0b1000001;
/// 		const __m256i permuted_data = _mm256_permutevar4x64_pd(data, shuffle);
/// \param mask
/// \return
const __m256i shuffle_up_64(const uint64_t mask) noexcept {
	ASSERT(mask < (1u << 4u));

	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	expanded_mask *= 0xFFU;
	const uint64_t identity_indices = 0x03020100;
	uint64_t wanted_indices = _pdep_u64(identity_indices, expanded_mask);

	const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
	const __m256i shufmask = _mm256_cvtepu8_epi64(bytevec);
	return shufmask;
}

/// similar to `shuffle_up_64`, but instead it can shuffle up to 8 64bit
///	limbs in parallel. Therefore it needs to return 2 __m256i
/// \param mask
const void shuffle_up_2_64(__m256i &higher, __m256i &lower, const uint64_t mask) noexcept {
	ASSERT(mask < (1u << 8u));

	uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
	expanded_mask *= 0xFFU;
	const uint64_t identity_indices = 0x03020100;
	uint64_t wanted_indices1 = _pdep_u64(identity_indices, expanded_mask & ((1ul << 32u) - 1));
	uint64_t wanted_indices2 = _pdep_u64(identity_indices, expanded_mask >> 32u);

	const __m128i bytevec1 = _mm_cvtsi32_si128(wanted_indices1);
	const __m128i bytevec2 = _mm_cvtsi32_si128(wanted_indices2);
	lower = _mm256_cvtepu8_epi64(bytevec1);
	higher = _mm256_cvtepu8_epi64(bytevec2);
}
#endif//DECODING_SHUFFLE_H
