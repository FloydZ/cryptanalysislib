#ifndef CRYPTANALYSISLIB_POPCOUNT_X86_H
#define CRYPTANALYSISLIB_POPCOUNT_X86_H

#include "helper.h"

struct popcount_config {
public:
	// TODO move this into custom alignment class
	// alignment in bytes
	const uint32_t alignment;

	constexpr popcount_config(const uint32_t alignment) noexcept :
		alignment(alignment) {}
};

class popcount {
private:
	constexpr static popcount_config config{16};

public:
	/// delete the default construct/deconstructor
	popcount() = delete;
	~popcount() = delete;


	///
	/// \tparam T
	/// \param data
	/// \return
	template<class T>
		requires std::integral<T>
	constexpr static uint32_t count(T data) noexcept {
		if constexpr(sizeof(T) < 8) {
			return __builtin_popcount(data);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_popcountll(data);
		} else {
			throw ;
		}
	}

	///
	/// \tparam T
	/// \param data
	/// \param size
	/// \return
	template<class T, popcount_config config = config>
		requires std::integral<T>
	constexpr static uint32_t count(T *__restrict__ data, const size_t size) noexcept {
		// ASSERT(uint64_t(data) % config.alignment);
		uint32_t sum = 0;
		for (size_t i = 0; i < size; ++i) {
			sum += count(data[i]);
		}

		return sum;
	}
};

/// special popcount which popcounts on 8 * 32u bit limbs in parallel
static __m256i popcount_avx2_32(const __m256i vec) noexcept {
	const __m256i lookup = _mm256_setr_epi8(
	    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
	    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
	);

	const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo  = _mm256_and_si256(vec, low_mask);
    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i local = _mm256_setzero_si256();
    local = _mm256_add_epi8(local, popcnt1);
    local = _mm256_add_epi8(local, popcnt2);

	// not the best
	const __m256i mask = _mm256_set1_epi32(0xff);
	__m256i ret = _mm256_and_si256(local, mask);
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local,  8), mask));
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 16), mask));
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 24), mask));
	return ret;
}

/// special popcount which popcounts on 4 * 64 bit limbs in parallel
static __m256i popcount_avx2_64(const __m256i vec) noexcept {
	const __m256i lookup = _mm256_setr_epi8(
	    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
	    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
	    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
	    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
	    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
	);

	const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i lo  = _mm256_and_si256(vec, low_mask);
    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i local = _mm256_setzero_si256();
    local = _mm256_add_epi8(local, popcnt1);
    local = _mm256_add_epi8(local, popcnt2);

	const __m256i mask2 = _mm256_set1_epi64x(0xff);
	__m256i ret;
	
	ret = _mm256_add_epi8(local, _mm256_srli_epi32(local,  8));
	ret = _mm256_add_epi8(ret, _mm256_srli_epi32(ret,  16));
	ret = _mm256_add_epi8(ret, _mm256_srli_epi64(ret,  32));
	ret = _mm256_and_si256(ret, mask2);
	return ret;
}


#endif
