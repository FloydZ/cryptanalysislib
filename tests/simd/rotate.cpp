#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `ror`/`rol` against a scalar rotation, for all rotation counts < 2 * bits
template<typename S>
static void check_rotate() {
	using T = typename S::limb_type;
	constexpr uint32_t bits = sizeof(T) * 8u;
	alignas(64) T a[S::LIMBS], o[S::LIMBS];
	for (uint32_t r = 0; r < 100; ++r) {
		for (uint32_t i = 0; i < S::LIMBS; ++i) { a[i] = T(rng()); }
		const S va = S::load(a);
		for (uint32_t k = 0; k < 2u * bits; ++k) {
			const uint32_t s = k % bits;
			S::store(o, S::ror(va, k));
			for (uint32_t i = 0; i < S::LIMBS; ++i) {
				const T e = s ? T((a[i] >> s) | (a[i] << (bits - s))) : a[i];
				EXPECT_EQ(o[i], e);
			}

			S::store(o, S::rol(va, k));
			for (uint32_t i = 0; i < S::LIMBS; ++i) {
				const T e = s ? T((a[i] << s) | (a[i] >> (bits - s))) : a[i];
				EXPECT_EQ(o[i], e);
			}
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(Rotate, simd128) {
	check_rotate<_uint8x16_t>();
	check_rotate<_uint16x8_t>();
	check_rotate<_uint32x4_t>();
	check_rotate<_uint64x2_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
