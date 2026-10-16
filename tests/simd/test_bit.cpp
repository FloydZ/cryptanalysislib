#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `test(in, bit_pos)`: bit i of the mask is bit `bit_pos` of `in[i]`
template<typename S>
static void check_test_bit() {
	using T = typename S::limb_type;
	using U = std::make_unsigned_t<T>;
	constexpr uint32_t N = S::LIMBS, B = sizeof(T) * 8u;
	alignas(64) T a[N];
	for (uint32_t r = 0; r < 200; ++r) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		const S va = S::load(a);
		for (uint32_t bp = 0; bp < B; ++bp) {
			uint64_t e = 0;
			for (uint32_t i = 0; i < N; ++i) { e |= uint64_t((U(a[i]) >> bp) & 1u) << i; }
			EXPECT_EQ(uint64_t(MaskType<N>(S::test(va, bp))), e);
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(TestBit, simd256) {
	check_test_bit<uint8x32_t>();
	check_test_bit<uint16x16_t>();
	check_test_bit<uint32x8_t>();
	check_test_bit<uint64x4_t>();
	check_test_bit<int8x32_t>();
	check_test_bit<int16x16_t>();
	check_test_bit<int32x8_t>();
	check_test_bit<int64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(TestBit, simd512) {
	check_test_bit<uint8x64_t>();
	check_test_bit<uint16x32_t>();
	check_test_bit<uint32x16_t>();
	check_test_bit<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
