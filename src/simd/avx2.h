#pragma once

#ifndef USE_AVX2
#error "no avx"
#endif

#include <immintrin.h>
#include <cstdint>
#include <cstdio>

#include "helper.h"

struct uint8x32_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	inline uint8x32_t random();
	inline uint8x32_t print();
};

struct uint16x16_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
	
	inline uint16x16_t random();
};


struct uint32x8_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
	
	inline uint32x8_t random();
};

struct uint64x4_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	inline uint64x4_t random();
};


inline uint8x32_t add(const uint8x32_t in1, 
					  const uint8x32_t in2) {
	uint8x32_t out;
	out.v256 = _mm256_add_epi8(in1.v256, in2.v256);
	return out;
}

inline uint16x16_t add(const uint16x16_t in1, 
					  const uint16x16_t in2) {
	uint16x16_t out;
	out.v256 = _mm256_add_epi16(in1.v256, in2.v256);
	return out;
}

inline uint32x8_t add(const uint32x8_t in1, 
					  const uint32x8_t in2) {
	uint32x8_t out;
	out.v256 = _mm256_add_epi32(in1.v256, in2.v256);
	return out;
}

inline uint64x4_t add(const uint64x4_t in1, 
					  const uint64x4_t in2) {
	uint64x4_t out;
	out.v256 = _mm256_add_epi64(in1.v256, in2.v256);
	return out;
}


inline uint8x32_t sub(const uint8x32_t in1, 
					  const uint8x32_t in2) {
	uint8x32_t out;
	out.v256 = _mm256_sub_epi8(in1.v256, in2.v256);
	return out;
}

inline uint16x16_t sub(const uint16x16_t in1, 
					   const uint16x16_t in2) {
	uint16x16_t out;
	out.v256 = _mm256_sub_epi16(in1.v256, in2.v256);
	return out;
}

inline uint32x8_t sub(const uint32x8_t in1, 
					  const uint32x8_t in2) {
	uint32x8_t out;
	out.v256 = _mm256_sub_epi32(in1.v256, in2.v256);
	return out;
}

inline uint64x4_t sub(const uint64x4_t in1, 
					  const uint64x4_t in2) {
	uint64x4_t out;
	out.v256 = _mm256_sub_epi64(in1.v256, in2.v256);
	return out;
}


inline uint8x32_t mullo(const uint8x32_t in1, 
					    const uint8x32_t in2) {
	uint8x32_t out;
	const __m256i maskl = _mm256_set1_epi8(0x0f);
	const __m256i maskh = _mm256_set1_epi8(0xf0);

	const __m256i in1l = _mm256_and_si256(in1.v256, maskl);
	const __m256i in2l = _mm256_srli_epi16(_mm256_and_si256(in2.v256, maskl), 8u);
	const __m256i in1h = _mm256_and_si256(in1.v256, maskh);
	const __m256i in2h = _mm256_srli_epi16(_mm256_and_si256(in2.v256, maskh), 8u);

	out.v256 = _mm256_mullo_epi16(in1l, in2l);
	const __m256i tho = _mm256_slli_epi16(_mm256_mullo_epi16(in1l, in2l), 8u);
	out.v256 = _mm256_xor_si256(tho, out.v256);
	return out;
}

inline uint16x16_t mullo(const uint16x16_t in1, 
					    const uint16x16_t in2) {
	uint16x16_t out;
	out.v256 = _mm256_mullo_epi16(in1.v256, in2.v256);
	return out;
}

inline uint32x8_t mullo(const uint32x8_t in1, 
					    const uint32x8_t in2) {
	uint32x8_t out;
	out.v256 = _mm256_mullo_epi16(in1.v256, in2.v256);
	return out;
}

inline uint64x4_t mullo(const uint64x4_t in1, 
					    const uint64x4_t in2) {
	ASSERT(false);
	uint64x4_t out;
	return out;
}




/// wrapper class/struct arounf __m256i for better debugging
union U256i {
    __m256i v;
    uint32_t a[8];
    uint64_t b[4];
};

/// prints a `__m256i` as 8 `u32`
void print_m256i_u32(const __m256i v){
    const U256i u = { v };

    for (uint32_t i = 0; i < 8; ++i) {
        printf("%d ", u.a[i]);

	}
	
	printf("\n");
}

