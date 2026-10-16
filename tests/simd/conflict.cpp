#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `conflict`: lane i is the bitmask of all lanes j < i with in[j] == in[i]
template<typename S>
static void check_conflict() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		// few distinct values, so that many lanes collide
		const uint32_t q = 1u + (r % 8u);
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng() % q); }

		S::store(o, S::conflict(S::load(a)));
		for (uint32_t i = 0; i < N; ++i) {
			uint64_t e = 0;
			for (uint32_t j = 0; j < i; ++j) { e |= uint64_t(a[j] == a[i]) << j; }
			EXPECT_EQ(uint64_t(o[i]), e);
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(Conflict, simd256) {
	check_conflict<uint16x16_t>();
	check_conflict<uint32x8_t>();
	check_conflict<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(Conflict, simd512) {
	check_conflict<uint32x16_t>();
	check_conflict<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
