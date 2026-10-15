#include <gtest/gtest.h>
#include <cstdint>
#include <map>

#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

TEST(random, rng_weighted_weight) {
	for (uint32_t r = 0; r < 1000; r++) {
		for (uint32_t w = 0; w < 64; w++) {
			EXPECT_EQ(__builtin_popcountll(rng_weighted<uint64_t>(w)), w);
		}
		for (uint32_t w = 0; w < 8; w++) {
			EXPECT_EQ(__builtin_popcount(rng_weighted<uint8_t>(w)), w);
		}
	}
}

TEST(random, rng_weighted_uniform) {
	// all 28 values of weight 2 in 8 bits, about equally often
	constexpr uint32_t N = 28 * 4000;
	std::map<uint8_t, uint32_t> cnt;
	for (uint32_t r = 0; r < N; r++) { cnt[rng_weighted<uint8_t>(2)]++; }
	EXPECT_EQ(cnt.size(), 28u);
	for (const auto &[k, v] : cnt) {
		EXPECT_GT(v, 3400u);
		EXPECT_LT(v, 4600u);
	}
}

TEST(random, signed_ranges) {
	for (uint32_t i = 0; i < 10000; i++) {
		const int32_t a = rng<int32_t>(-5, 5);
		EXPECT_GE(a, -5); EXPECT_LT(a, 5);
		const int8_t b = rng<int8_t>(int8_t(10));
		EXPECT_GE(b, 0); EXPECT_LT(b, 10);
		const int64_t c = rng<int64_t>(INT64_MIN, INT64_MAX);
		EXPECT_LT(c, INT64_MAX);
	}
}

TEST(random, random_device_seed) {
	random_device d(0);
	random_device e(42);
	(void) d(); (void) e();
	random_device f(std::string("abc"));
	(void) f();
}

TEST(random, xorshf96_seed) {
	EXPECT_TRUE(cryptanalysislib::random::internal::xorshf96_seed());
}

TEST(random, rng_seed_reproducible) {
	constexpr uint32_t N = 16;
	for (const uint64_t seed : {uint64_t(0), uint64_t(1), uint64_t(42), uint64_t(-123456789)}) {
		uint64_t a[N], b[N];
		rng_seed(seed);
		uint32_t zeros = 0;
		for (uint32_t i = 0; i < N; i++) { a[i] = rng(); zeros += a[i] == 0; }
		EXPECT_LT(zeros, N);

		(void) rng(); // advance the state
		rng_seed(seed);
		for (uint32_t i = 0; i < N; i++) { b[i] = rng(); }
		for (uint32_t i = 0; i < N; i++) { EXPECT_EQ(a[i], b[i]); }
	}
}

TEST(random, pcg64_unseeded) {
	uint32_t zeros = 0;
	for (uint32_t i = 0; i < 16; i++) {
		zeros += cryptanalysislib::random::internal::pcg64_random_data() == 0;
	}
	EXPECT_EQ(zeros, 0u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
