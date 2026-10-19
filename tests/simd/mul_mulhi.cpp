#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `mul` (lower half) and `mulhi` (upper half) of the 32x32 -> 64 bit products
template<typename S>
static void check_mul_mulhi() {
	using T = typename S::limb_type;
	using W = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], b[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); b[i] = T(rng()); }
		if (r == 0) { for (uint32_t i = 0; i < N; ++i) { a[i] = T(-1); b[i] = T(-1 - int(i)); } }

		const S va = S::load(a), vb = S::load(b);
		S::store(o, S::mul(va, vb));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(W(a[i]) * W(b[i]))); }

		S::store(o, S::mulhi(va, vb));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T((W(a[i]) * W(b[i])) >> 32u)); }
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(MulMulhi, simd256) {
	check_mul_mulhi<uint32x8_t>();
	check_mul_mulhi<int32x8_t>();
}
#endif

#ifdef USE_AVX512F
TEST(MulMulhi, simd512) {
	check_mul_mulhi<uint32x16_t>();
	check_mul_mulhi<int32x16_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
