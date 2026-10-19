#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `gt`, `ge`, `lt` and `le` (compressed and uncompressed) against the
/// scalar comparisons, for signed and unsigned types
template<typename S>
static void check_compare() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], b[N], o[N];
	const auto mask = [](const bool c) { return T(c ? -1 : 0); };
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			a[i] = T(rng());
			b[i] = (i % 3u == 0) ? a[i] : T(rng());
		}

		const S va = S::load(a), vb = S::load(b);
		const uint64_t gt = S::gt(va, vb), ge = S::ge(va, vb),
		               lt = S::lt(va, vb), le = S::le(va, vb);
		for (uint32_t i = 0; i < N; ++i) {
			EXPECT_EQ((gt >> i) & 1u, a[i] >  b[i]);
			EXPECT_EQ((ge >> i) & 1u, a[i] >= b[i]);
			EXPECT_EQ((lt >> i) & 1u, a[i] <  b[i]);
			EXPECT_EQ((le >> i) & 1u, a[i] <= b[i]);
		}

		S::store(o, S::gt_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] >  b[i])); }
		S::store(o, S::ge_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] >= b[i])); }
		S::store(o, S::lt_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] <  b[i])); }
		S::store(o, S::le_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] <= b[i])); }
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(Compare, simd128_unsigned) {
	check_compare<_uint8x16_t>();
	check_compare<_uint16x8_t>();
	check_compare<_uint32x4_t>();
	check_compare<_uint64x2_t>();
}

TEST(Compare, simd128_signed) {
	check_compare<_int8x16_t>();
	check_compare<_int16x8_t>();
	check_compare<_int32x4_t>();
	check_compare<_int64x2_t>();
}

TEST(Compare, simd256_unsigned) {
	check_compare<uint8x32_t>();
	check_compare<uint16x16_t>();
	check_compare<uint32x8_t>();
	check_compare<uint64x4_t>();
}

TEST(Compare, simd256_signed) {
	check_compare<int8x32_t>();
	check_compare<int16x16_t>();
	check_compare<int32x8_t>();
	check_compare<int64x4_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
