#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `reduce_min`/`reduce_max` against `std::min_element`/`std::max_element`
template<typename S>
static void check_reduce_minmax() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		const S va = S::load(a);
		EXPECT_EQ(S::reduce_min(va), *std::min_element(a, a + N));
		EXPECT_EQ(S::reduce_max(va), *std::max_element(a, a + N));
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(ReduceMinMax, simd256) {
	check_reduce_minmax<uint8x32_t>();
	check_reduce_minmax<uint16x16_t>();
	check_reduce_minmax<uint32x8_t>();
	check_reduce_minmax<uint64x4_t>();
	check_reduce_minmax<int8x32_t>();
	check_reduce_minmax<int16x16_t>();
	check_reduce_minmax<int32x8_t>();
	check_reduce_minmax<int64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(ReduceMinMax, simd512) {
	check_reduce_minmax<uint8x64_t>();
	check_reduce_minmax<uint16x32_t>();
	check_reduce_minmax<uint32x16_t>();
	check_reduce_minmax<uint64x8_t>();
	check_reduce_minmax<int8x64_t>();
	check_reduce_minmax<int16x32_t>();
	check_reduce_minmax<int32x16_t>();
	check_reduce_minmax<int64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
