#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#include <cstdint>
#include <cinttypes>

#include "helper.h"
#include "random.h"


#if defined(USE_AVX2)

#include "simd/avx2.h"

#elif defined(USE_NEON)

#include "simd/neon.h"

#elif defined(USE_RISCV)

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
constexpr inline void uint8x32_t::print(bool binary, bool hex){
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 32; i++) {
			printbinary(this->v8[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 32; i++) {
			printf("%hhx ", this->v8[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 32; i++) {
		printf("%u ", this->v8[i]);
	}
	printf("\n");
}

inline void uint16x16_t::print(bool binary, bool hex){
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 16; i++) {
			printbinary(this->v16[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 16; i++) {
			printf("%hx ", this->v16[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 16; i++) {
		printf("%u ", this->v16[i]);
	}
	printf("\n");
}

inline void uint32x8_t::print(bool binary, bool hex){
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 8; i++) {
			printbinary(this->v32[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 8; i++) {
			printf("%x ", this->v32[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 8; i++) {
		printf("%u ", this->v32[i]);
	}
	printf("\n");
}

inline void uint64x4_t::print(bool binary, bool hex){
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 4; i++) {
			printbinary(this->v64[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 4; i++) {
			printf("%lx ", this->v64[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 4; i++) {
		printf("%" PRIu64 " ", this->v64[i]);
	}
	printf("\n");
}

#endif//CRYPTANALYSISLIB_SIMD_H
