#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// masked loads: lanes not in the mask are zero (random and special masks)
template<typename S>
static void check_maskz_load() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	constexpr uint64_t all = (N == 64) ? ~0ull : ((1ull << N) - 1ull);
	alignas(64) T a[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		uint64_t m = rng() & all;
		if (r == 0) { m = 0; }
		if (r == 1) { m = all; }
		if (r >= 2 && r < 2 + N) { m = 1ull << (r - 2); }

		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		const Mask<N> mask = Mask<N>(MaskType<N>(m));
		S::store(o, S::maskz_unaligned_load(a, mask));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], ((m >> i) & 1u) ? a[i] : T(0)); }
		S::store(o, S::maskz_aligned_load(a, mask));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], ((m >> i) & 1u) ? a[i] : T(0)); }
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(MaskzLoad, simd256) {
	check_maskz_load<uint8x32_t>();
	check_maskz_load<uint16x16_t>();
	check_maskz_load<uint32x8_t>();
	check_maskz_load<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(MaskzLoad, simd512) {
	check_maskz_load<uint8x64_t>();
	check_maskz_load<uint16x32_t>();
	check_maskz_load<uint32x16_t>();
	check_maskz_load<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
