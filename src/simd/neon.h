#ifndef CRYPTANALYSISLIB_SIMD_NEON_H
#define CRYPTANALYSISLIB_SIMD_NEON_H
#include <arm_neon.h>
#include <cstdint>

struct uint8x32_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
};

struct uint16x16_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
};

struct uint32x8_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
};

struct uint64x4_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
};

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
