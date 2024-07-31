#include <gtest/gtest.h>
#include <iostream>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#define S uint8x32_t
#include "test_simd.h"
#undef S
#define S uint16x16_t
#include "test_simd.h"
#undef S
#define S uint32x8_t
#include "test_simd.h"
#undef S
#define S uint64x4_t
#include "test_simd.h"
#undef S

#ifdef USE_AVX512F
#define S uint8x64_t
#include "test_simd.h"
#undef S
#define S uint16x32_t
#include "test_simd.h"
#undef S
#define S uint32x16_t
#include "test_simd.h"
#undef S
#define S uint64x8_t
#include "test_simd.h"
#undef S
#endif


TEST(uint32x8_t, set) {
	uint32_t pos = 21;
	uint8x32_t t1 = uint8x32_t::set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (uint32_t i = 0; i < 32; ++i) {
		if (i == pos) {
			EXPECT_EQ(t1.v8[i], 1);
			continue;
		}
		EXPECT_EQ(t1.v8[i], 0);
	}

	uint8x32_t t2 = uint8x32_t::setr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (uint32_t i = 0; i < 32; ++i) {
		if (i == (31 - pos)) {
			EXPECT_EQ(t2.v8[i], 1);
			continue;
		}
		EXPECT_EQ(t2.v8[i], 0);
	}
}

TEST(uint8x32_t, slri) {
	for (uint8_t j = 0; j < 8; j++) {
		const uint8x32_t t1 = uint8x32_t::set1(1u << j);
		const uint8x32_t t2 = uint8x32_t::srli(t1, j);
		for (uint32_t i = 0; i < 32; ++i) {
			EXPECT_EQ(t2.v8[i], 1);
		}
	}

	/// special case for j = 8
	const uint8x32_t t1 = uint8x32_t::set1((1u << 7u) - 1u);
	const uint8x32_t t2 = uint8x32_t::srli(t1, 8);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t2.v8[i], 0);
	}
}


#ifdef USE_AVX2
TEST(avx, prefixsum) {
	constexpr size_t s = 65;
	uint32_t *d1 = (uint32_t *)malloc(s * sizeof(uint32_t));
	uint32_t *d2 = (uint32_t *)malloc(s * sizeof(uint32_t));
	// for (uint32_t i = 0; i < s; i++) { d1[i] = fastrandombytes_uint64() % (1u << 8u); }
	for (uint32_t i = 0; i < s; i++) {
		d1[i] = i;
	}
	memcpy(d2, d1, s * sizeof(uint32_t));

	avx2_prefixsum_u32(d1, s);
   	for (uint32_t i = 1; i < s; i++) {
        d2[i] += d2[i - 1];
	}

	for (uint32_t i = 0; i < s; i++) {
		EXPECT_EQ(d1[i], d2[i]);
	}


	free(d1); free(d2);
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
