#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#include <cstdint>
#include "random.h"

#ifdef USE_AVX2

#include "simd/avx2.h"

#elifdef USE_NEON

#include "simd/neon.h"

#elifdef USE_RISCV

#include "simd/riscv.h"

#else

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

///
inline uinuint8x32_t add(uint8x32_t out, 
						 const uint8x32_t in1, 
						 const uint8x32_t in2) {
	for (uint32_t i = 0; i < 32; i++) {
		out.v8[i] = in1.v8[i] + in2.v8[i];
	}
}
#endif // no SIMD uinit available



/// functions which are shared among all implementations.
inline uint8x32_t uint8x32_t::random(){
	uint8x32_t ret;
	for (uint32_t i = 0; i < 4; i++) {
		ret.v64[i] = fastrandombytes_uint64();
	}

	return ret;
}
inline uint16x16_t uint16x16_t::random(){
	uint16x16_t ret;
	for (uint32_t i = 0; i < 4; i++) {
		ret.v64[i] = fastrandombytes_uint64();
	}

	return ret;
}
inline uint32x8_t uint32x8_t::random(){
	uint32x8_t ret;
	for (uint32_t i = 0; i < 4; i++) {
		ret.v64[i] = fastrandombytes_uint64();
	}

	return ret;
}
inline uint64x4_t uint64x4_t::random(){
	uint64x4_t ret;
	for (uint32_t i = 0; i < 4; i++) {
		ret.v64[i] = fastrandombytes_uint64();
	}

	return ret;
}

inline uint8x32_t uint8x32_t::print(){
	uint8x32_t ret;
	for (uint32_t i = 0; i < 32; i++) {
		printf("%u ", this->v8[i]);
	}
	printf("\n");

	return ret;
}
#endif//CRYPTANALYSISLIB_SIMD_H
