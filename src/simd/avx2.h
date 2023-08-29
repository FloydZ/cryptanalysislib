#pragma once

#ifndef USE_AVX2
#error "no avx"
#endif

#include <immintrin.h>
#include <cstdint>
#include <cstdio>

#include "helper.h"

//
#if CUSTOM_ALIGNMENT
#define MM256_LOAD(x) _mm256_load_ps(x)
#define MM256_STORE(x, y) _mm256_store_ps(x, y)
#else
#define MM256_LOAD(x) _mm256_loadu_ps(x)
#define MM256_STORE(x, y) _mm256_storeu_ps(x, y)
#endif


#define MM256_LOAD_UNALIGNED(x) _mm256_loadu_ps(x)
#define MM256_STORE_UNALIGNED(x, y) _mm256_storeu_ps(x, y)

// Load instruction which do not touch the cache
#define MM256_STREAM_LOAD256(x) _mm256_stream_load_si256(x)
#define MM256_STREAM_LOAD128(x) _mm_stream_load_si128(x)
#define MM256_STREAM_LOAD64(x)  __asm volatile("movntdq %0, (%1)\n\t"       \
												:                           \
												: "x" (x), "r" (y)          \
												: "memory");
// Write instructions which do not touch the cache
#define MM256_STREAM256(x, y) _mm256_stream_load_si256(x, y)
#define MM256_STREAM128(x, y) _mm_stream_si128(x)
#define MM256_STREAM64(x, y)  __asm volatile("movntdq %0, (%1)\n\t"         \
												:                           \
												: "x" (y), "r" (x)          \
												: "memory");


struct uint8x32_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint8x32_t random(){
		uint8x32_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint8x32_t set1(const uint8_t a) {
		uint8x32_t out;
		out.v256 = _mm256_set1_epi8(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint8x32_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint8x32_t aligned_load(const void *ptr) {
		uint8x32_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint8x32_t unaligned_load(const void *ptr) {
		uint8x32_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint8x32_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint8x32_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint8x32_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t xor_(const uint8x32_t in1,
	                              const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t and_(const uint8x32_t in1,
	                              const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t or_(const uint8x32_t in1,
						  const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t andnot(const uint8x32_t in1,
	                                const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	static inline uint8x32_t not_(const uint8x32_t in1) {
		uint8x32_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t add(const uint8x32_t in1,
	                             const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = _mm256_add_epi8(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t sub(const uint8x32_t in1,
	                             const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = _mm256_sub_epi8(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t mullo(const uint8x32_t in1,
	                               const uint8x32_t in2) {
		uint8x32_t out;
		const __m256i maskl = _mm256_set1_epi16(0x000f);
		const __m256i maskh = _mm256_set1_epi16(0x0f00);

		const __m256i in1l = _mm256_and_si256(in1.v256, maskl);
		const __m256i in2l = _mm256_and_si256(in2.v256, maskl);
		const __m256i in1h = _mm256_srli_epi16(_mm256_and_si256(in1.v256, maskh), 8u);
		const __m256i in2h = _mm256_srli_epi16(_mm256_and_si256(in2.v256, maskh), 8u);

		out.v256 = _mm256_mullo_epi16(in1l, in2l);
		const __m256i tho = _mm256_slli_epi16(_mm256_mullo_epi16(in1h, in2h), 8u);
		out.v256 = _mm256_xor_si256(tho, out.v256);
		return out;
	}
};

struct uint16x16_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint16x16_t random(){
		uint16x16_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint16x16_t set1(const uint16_t a) {
		uint16x16_t out;
		out.v256 = _mm256_set1_epi16(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint16x16_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint16x16_t aligned_load(const void *ptr) {
		uint16x16_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint16x16_t unaligned_load(const void *ptr) {
		uint16x16_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint16x16_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint16x16_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint16x16_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t xor_(const uint16x16_t in1,
							const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t and_(const uint16x16_t in1,
							const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t or_(const uint16x16_t in1,
						   const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t andnot(const uint16x16_t in1,
							  const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	inline uint16x16_t not_(const uint16x16_t in1) {
		uint16x16_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t add(const uint16x16_t in1,
						   const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_add_epi16(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t sub(const uint16x16_t in1,
						   const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_sub_epi16(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t mullo(const uint16x16_t in1,
							 const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_mullo_epi16(in1.v256, in2.v256);
		return out;
	}
};

struct uint32x8_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint32x8_t random(){
		uint32x8_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint32x8_t set1(const uint32_t a) {
		uint32x8_t out;
		out.v256 = _mm256_set1_epi32(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint32x8_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint32x8_t aligned_load(const void *ptr) {
		uint32x8_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint32x8_t unaligned_load(const void *ptr) {
		uint32x8_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint32x8_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint32x8_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint32x8_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t xor_(const uint32x8_t in1,
						   const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t and_(const uint32x8_t in1,
						   const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t or_(const uint32x8_t in1,
						  const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t andnot(const uint32x8_t in1,
							 const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	inline uint32x8_t not_(const uint32x8_t in1) {
		uint32x8_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t add(const uint32x8_t in1,
						  const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_add_epi32(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t sub(const uint32x8_t in1,
						  const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_sub_epi32(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t mullo(const uint32x8_t in1,
							const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_mullo_epi16(in1.v256, in2.v256);
		return out;
	}
};

struct uint64x4_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint64x4_t random(){
		uint64x4_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint64x4_t set1(const uint64_t a) {
		uint64x4_t out;
		out.v256 = _mm256_set1_epi64x(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint64x4_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t aligned_load(const void *ptr) {
		uint64x4_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t unaligned_load(const void *ptr) {
		uint64x4_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint64x4_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint64x4_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint64x4_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t xor_(const uint64x4_t in1,
						   const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t and_(const uint64x4_t in1,
						   const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t or_(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t andnot(const uint64x4_t in1,
							 const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	inline uint64x4_t not_(const uint64x4_t in1) {
		uint64x4_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t add(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_add_epi64(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t sub(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_sub_epi64(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t mullo(const uint64x4_t in1,
							const uint64x4_t in2) {
		ASSERT(false);
		uint64x4_t out;
		return out;
	}
};

bool operator==(const uint8x32_t& a, const uint8x32_t& b){
	const __m256i tmp = _mm256_cmpeq_epi8(a.v256, b.v256);
	const int mask = _mm256_movemask_epi8(tmp);
	return mask == 0xffffffff;
}

