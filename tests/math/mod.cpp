#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "math/math.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


using namespace cryptanalysislib;

TEST(fastmod, rand) {
    using TypeParam = uint32_t;
    TypeParam a = 5;
	for (uint32_t i = 0; i < 100; ++i) {
        a = next_prime(a);
	}
}

TEST(fastmod, u32) {
	using namespace cryptanalysislib::math::internal;
	for (const uint32_t d : {2u, 3u, 7u, 1000u, 65537u, 0x7FFFFFFFu, 0xFFFFFFFFu}) {
		const uint64_t M = computeM_u32(d);
		for (uint32_t i = 0; i < 10000; ++i) {
			const uint32_t a = rng<uint32_t>();
			EXPECT_EQ(fastmod_u32(a, M, d), a % d);
			EXPECT_EQ(fastdiv_u32(a, M), a / d);
		}
	}

	EXPECT_EQ(fastmod<7u>(100u), 2u);
	EXPECT_EQ(fastdiv<7u>(100u), 14u);
}

TEST(fastmod, u64) {
	using namespace cryptanalysislib::math::internal;
	for (const uint64_t d : {2ull, 3ull, 7ull, 1000ull, (1ull << 32) + 15ull, 0x7FFFFFFFFFFFFFFFull, ~0ull}) {
		const __uint128_t M = computeM_u64(d);
		for (uint32_t i = 0; i < 10000; ++i) {
			const uint64_t a = rng<uint64_t>();
			EXPECT_EQ(fastmod_u64(a, M, d), a % d);
			EXPECT_EQ(fastdiv_u64(a, M), a / d);
			EXPECT_EQ(is_divisible_u64(a - (a % d), M), true);
		}
	}

	constexpr uint64_t x = (1ull << 40) + 12345ull;
	static_assert(fastmod<1000003ull>(x) == x % 1000003ull);
	static_assert(fastdiv<1000003ull>(x) == x / 1000003ull);
	static_assert(fastdiv<1ull>(x) == x);
}

TEST(fastmod, simd_u32) {
	using namespace cryptanalysislib::math::internal;
	using S = uint64x4_t;
	for (const uint32_t d : {2u, 3u, 7u, 1000u, 65537u, 0xFFFFFFFFu}) {
		const uint64_t M = computeM_u32(d);
		const S sM = S::set1(M), sd = S::set1(d);
		for (uint32_t i = 0; i < 1000; ++i) {
			uint64_t a[4];
			for (uint32_t j = 0; j < 4; ++j) { a[j] = rng<uint32_t>(); }
			const S sa = S::load(a);
			const S m = fastmod_u32(sa, sM, sd);
			const S q = fastdiv_u32(sa, sM);
			for (uint32_t j = 0; j < 4; ++j) {
				EXPECT_EQ(m[j], a[j] % d);
				EXPECT_EQ(q[j], a[j] / d);
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
