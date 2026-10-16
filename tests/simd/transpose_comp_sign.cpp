#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `transpose`: bit i of output byte j is bit j of input byte (7 - i), in
/// each 64 bit word; `comp_sign` against the scalar `psignb`
template<typename S>
static void check_transpose_comp_sign() {
	constexpr uint32_t N = S::LIMBS;
	alignas(64) uint8_t a[N], b[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			a[i] = uint8_t(rng());
			b[i] = (i % 4u == 0) ? 0 : uint8_t(rng());
		}
		const S va = S::load(a), vb = S::load(b);

		S::store(o, S::transpose(va));
		for (uint32_t w = 0; w < N; w += 8) {
			for (uint32_t j = 0; j < 8; ++j) {
				for (uint32_t i = 0; i < 8; ++i) {
					EXPECT_EQ((o[w + j] >> i) & 1u, (a[w + 7 - i] >> j) & 1u);
				}
			}
		}

		S::store(o, S::comp_sign(va, vb));
		for (uint32_t i = 0; i < N; ++i) {
			const int8_t bi = int8_t(b[i]);
			const uint8_t e = bi == 0 ? 0 : (bi < 0 ? uint8_t(-a[i]) : a[i]);
			EXPECT_EQ(o[i], e);
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(TransposeCompSign, simd256) {
	check_transpose_comp_sign<uint8x32_t>();
}
#endif

#ifdef USE_AVX512F
TEST(TransposeCompSign, simd512) {
	check_transpose_comp_sign<uint8x64_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
