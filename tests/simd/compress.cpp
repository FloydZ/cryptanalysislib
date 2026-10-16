#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// the three `compress` overloads against a scalar reference
template<typename S>
static void check_compress() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	constexpr uint64_t all = (N == 64) ? ~0ull : ((1ull << N) - 1ull);
	alignas(64) T a[N], b[N], o[N], p[N + 1];
	for (uint32_t r = 0; r < 1000; ++r) {
		uint64_t m = rng() & all;
		if (r == 0) { m = 0; }
		if (r == 1) { m = all; }

		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); b[i] = T(rng()); }
		const Mask<N> mask = Mask<N>(MaskType<N>(m));
		const S va = S::load(a), vb = S::load(b);

		// reference
		T e[N];
		uint32_t k = 0;
		for (uint32_t i = 0; i < N; ++i) {
			if ((m >> i) & 1u) { e[k++] = a[i]; }
		}

		S::store(o, S::compress(va, mask));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], i < k ? e[i] : T(0)); }

		S::store(o, S::compress(va, vb, mask));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], i < k ? e[i] : b[i]); }

		for (uint32_t i = 0; i < N + 1; ++i) { p[i] = T(0x5a); }
		S::compress(p, va, mask);
		for (uint32_t i = 0; i < N + 1; ++i) { EXPECT_EQ(p[i], i < k ? e[i] : T(0x5a)); }
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(Compress, simd256) {
	check_compress<uint8x32_t>();
	check_compress<uint16x16_t>();
	check_compress<uint32x8_t>();
	check_compress<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(Compress, simd512) {
	check_compress<uint8x64_t>();
	check_compress<uint16x32_t>();
	check_compress<uint32x16_t>();
	check_compress<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
