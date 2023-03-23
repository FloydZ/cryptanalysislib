#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#ifdef __x86_64_
#include <immintrin.h>
using u64_2 = __m128;
#endif

#ifdef __APPLE__
#include <arm_neon.h>
using u64_2 = uint64x2_t;

u64_2 xorb(const u64_2 a, const u64_2 b) {
	return veorq_u64( a, b);
}

void print(const u64_2 a) {
	std::cout << a[0] << " " << a[1] << "\n";
}


#endif
#endif//CRYPTANALYSISLIB_SIMD_H
