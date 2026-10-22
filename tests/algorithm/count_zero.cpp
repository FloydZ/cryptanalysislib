#include <gtest/gtest.h>
#include <bit>
#include <cstdint>

#include "algorithm/countl.h"
#include "algorithm/countr.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

static_assert(cryptanalysislib::countr_zero(0u) == 32);
static_assert(cryptanalysislib::countl_zero(0u) == 32);
static_assert(cryptanalysislib::countr_zero((uint8_t)0) == 8);
static_assert(cryptanalysislib::countl_zero((uint8_t)1) == 7);
static_assert(cryptanalysislib::countl_zero((uint16_t)1) == 15);
static_assert(cryptanalysislib::countr_zero(1ull << 63u) == 63);
static_assert(cryptanalysislib::countl_zero((unsigned __int128)1) == 127);
static_assert(cryptanalysislib::countr_zero((unsigned __int128)0) == 128);

/// every single bit, 0, all ones and random values (with random low/high
/// zero bits) against `std::countr_zero/countl_zero`
template<typename T>
static void check() {
	constexpr uint32_t bits = sizeof(T) * 8u;
	auto cmp = [](const T x) {
		EXPECT_EQ(cryptanalysislib::countr_zero(x), (uint32_t)std::countr_zero(x)) << (uint64_t)x;
		EXPECT_EQ(cryptanalysislib::countl_zero(x), (uint32_t)std::countl_zero(x)) << (uint64_t)x;
		EXPECT_EQ(cryptanalysislib::bit_floor(x), std::bit_floor(x)) << (uint64_t)x;
	};

	cmp(T(0));
	cmp(T(~T(0)));
	for (uint32_t i = 0; i < bits; ++i) { cmp(T(T(1) << i)); }
	for (size_t i = 0; i < 10000; ++i) {
		const T x = (T)rng<uint64_t>();
		cmp(T(x << (i % bits)));
		cmp(T(x >> (i % bits)));
	}
}

TEST(count_zero, uint8)  { check<uint8_t>(); }
TEST(count_zero, uint16) { check<uint16_t>(); }
TEST(count_zero, uint32) { check<uint32_t>(); }
TEST(count_zero, uint64) { check<uint64_t>(); }

TEST(count_zero, uint128) {
	using T = unsigned __int128;
	EXPECT_EQ(cryptanalysislib::countr_zero(T(0)), 128u);
	EXPECT_EQ(cryptanalysislib::countl_zero(T(0)), 128u);
	for (uint32_t i = 0; i < 128; ++i) {
		const T x = T(1) << i;
		EXPECT_EQ(cryptanalysislib::countr_zero(x), i);
		EXPECT_EQ(cryptanalysislib::countl_zero(x), 127u - i);
		// additional higher/lower garbage bits must not change the result
		EXPECT_EQ(cryptanalysislib::countr_zero(T(x | (x << 1u))), i);
		EXPECT_EQ(cryptanalysislib::countl_zero(T(x | (x >> 1u))), 127u - i);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
