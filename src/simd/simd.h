#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#ifdef USE_AVX2
#include <immintrin.h>
using u64_2 = __m128;

#include "simd/avx2.h"
#endif

#ifdef USE_NEON
#include <arm_neon.h>
using u64_2 = uint64x2_t;

u64_2 xorb(const u64_2 a, const u64_2 b) {
	return veorq_u64( a, b);
}

u64_2 gather(const uint64_t *ptr, const u64_2 off) {
	return vld1q_lane_u64(ptr, off, 0);
}

void store(uint64_t *ptr, const u64_2 off) {
	vst1q_lane_u64(ptr, off, 0);
}

void print(const u64_2 a) {
	std::cout << a[0] << " " << a[1] << "\n";
}


#endif
#endif//CRYPTANALYSISLIB_SIMD_H
