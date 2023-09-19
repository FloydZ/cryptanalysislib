#include <gtest/gtest.h>
#include <iostream>
#include <immintrin.h>

#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t stack_size = 10;

#ifdef USE_AVX2
#include "simd/avx2.h"
#include "popcount/avx2.h"
TEST(AVX2, uint32_t) {
	__m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i b = popcount_avx2_32(a);

	U256i c = U256i {b};
	EXPECT_EQ(c.a[0], 0);
	EXPECT_EQ(c.a[1], 1);
	EXPECT_EQ(c.a[2], 1);
	EXPECT_EQ(c.a[3], 2);
	EXPECT_EQ(c.a[4], 1);
	EXPECT_EQ(c.a[5], 2);
	EXPECT_EQ(c.a[6], 2);
	EXPECT_EQ(c.a[7], 3);
}

TEST(AVX2, uint64_t) {
	__m256i a = _mm256_setr_epi64x(0, 1, 3, 7);
	__m256i b = popcount_avx2_64(a);

	U256i c = U256i {b};
	EXPECT_EQ(c.b[0], 0);
	EXPECT_EQ(c.b[1], 1);
	EXPECT_EQ(c.b[2], 2);
	EXPECT_EQ(c.b[3], 3);
}

TEST(AVX2, uint64_t_random) {
	for (size_t i = 0; i < (1u << 8u); i++) {
		const uint64_t r1 = fastrandombytes_uint64();
		const uint64_t r2 = fastrandombytes_uint64();
		const uint64_t r3 = fastrandombytes_uint64();
		const uint64_t r4 = fastrandombytes_uint64();
		__m256i a = _mm256_setr_epi64x(r1, r2, r3, r4);
		__m256i b = popcount_avx2_64(a);

		U256i c = U256i {b};
		EXPECT_EQ(c.b[0], __builtin_popcountll(r1));
		EXPECT_EQ(c.b[1], __builtin_popcountll(r2));
		EXPECT_EQ(c.b[2], __builtin_popcountll(r3));
		EXPECT_EQ(c.b[3], __builtin_popcountll(r4));
	}
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
