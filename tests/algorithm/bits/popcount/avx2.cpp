#include <gtest/gtest.h>
#include <iostream>

#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;


#ifdef USE_AVX2
#include "algorithm/bits/popcount.h"
#include "simd/simd.h"
#include <immintrin.h>

constexpr size_t stack_size = 10;

TEST(AVX2, uint8_t) {
	__m256i a = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
	                             0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
	__m256i b = popcount_avx2_8(a);

	uint32x8_t c{};
	c.v256 = b;
	EXPECT_EQ(c.v8[0], 0);
	EXPECT_EQ(c.v8[1], 1);
	EXPECT_EQ(c.v8[2], 1);
	EXPECT_EQ(c.v8[3], 2);
	EXPECT_EQ(c.v8[4], 1);
	EXPECT_EQ(c.v8[5], 2);
	EXPECT_EQ(c.v8[6], 2);
	EXPECT_EQ(c.v8[7], 3);
	EXPECT_EQ(c.v8[8], 0);
	EXPECT_EQ(c.v8[9], 1);
	EXPECT_EQ(c.v8[10], 1);
	EXPECT_EQ(c.v8[11], 2);
	EXPECT_EQ(c.v8[12], 1);
	EXPECT_EQ(c.v8[13], 2);
	EXPECT_EQ(c.v8[14], 2);
	EXPECT_EQ(c.v8[15], 3);
}

TEST(AVX2, uint16_t) {
	__m256i a = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
	__m256i b = popcount_avx2_16(a);

	uint32x8_t c{};
	c.v256 = b;
	EXPECT_EQ(c.v16[0], 0);
	EXPECT_EQ(c.v16[1], 1);
	EXPECT_EQ(c.v16[2], 1);
	EXPECT_EQ(c.v16[3], 2);
	EXPECT_EQ(c.v16[4], 1);
	EXPECT_EQ(c.v16[5], 2);
	EXPECT_EQ(c.v16[6], 2);
	EXPECT_EQ(c.v16[7], 3);
	EXPECT_EQ(c.v16[8], 0);
	EXPECT_EQ(c.v16[9], 1);
	EXPECT_EQ(c.v16[10], 1);
	EXPECT_EQ(c.v16[11], 2);
	EXPECT_EQ(c.v16[12], 1);
	EXPECT_EQ(c.v16[13], 2);
	EXPECT_EQ(c.v16[14], 2);
	EXPECT_EQ(c.v16[15], 3);
}

TEST(AVX2, uint32_t) {
	__m256i a = _mm256_setr_epi32(0, 0xffffffff, 2, 3, 4, 5, 6, 7);
	__m256i b = popcount_avx2_32(a);

	uint32x8_t c{};
	c.v256 = b;
	EXPECT_EQ(c.v32[0], 0);
	EXPECT_EQ(c.v32[1], 32);
	EXPECT_EQ(c.v32[2], 1);
	EXPECT_EQ(c.v32[3], 2);
	EXPECT_EQ(c.v32[4], 1);
	EXPECT_EQ(c.v32[5], 2);
	EXPECT_EQ(c.v32[6], 2);
	EXPECT_EQ(c.v32[7], 3);
}

TEST(AVX2, uint64_t) {
	__m256i a = _mm256_setr_epi64x(0, 1, 3, 7);
	__m256i b = popcount_avx2_64(a);

	uint32x8_t c{};
	c.v256 = b;
	EXPECT_EQ(c.v64[0], 0);
	EXPECT_EQ(c.v64[1], 1);
	EXPECT_EQ(c.v64[2], 2);
	EXPECT_EQ(c.v64[3], 3);
}

TEST(AVX2, uint64_t_random) {
	for (size_t i = 0; i < (1u << 8u); i++) {
		const uint64_t r1 = rng();
		const uint64_t r2 = rng();
		const uint64_t r3 = rng();
		const uint64_t r4 = rng();
		__m256i a = _mm256_setr_epi64x(r1, r2, r3, r4);
		__m256i b = popcount_avx2_64(a);

		uint32x8_t c{};
		c.v256 = b;
		EXPECT_EQ(c.v64[0], popcount::popcount(r1));
		EXPECT_EQ(c.v64[1], popcount::popcount(r2));
		EXPECT_EQ(c.v64[2], popcount::popcount(r3));
		EXPECT_EQ(c.v64[3], popcount::popcount(r4));
	}
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
