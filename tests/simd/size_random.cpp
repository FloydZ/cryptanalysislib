#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `size()`, `is_unsigned()` and `random()` of the unsigned type `U` and
/// the signed type `I`
template<typename U, typename I>
static void check_size_random() {
	static_assert(U::size() == U::LIMBS);
	static_assert(I::size() == I::LIMBS);
	static_assert(U::is_unsigned());
	static_assert(!I::is_unsigned());

	// two random 128 bit registers are equal with probability 2^-128
	const U a = U::random(), b = U::random();
	bool equal = true;
	for (uint32_t i = 0; i < U::LIMBS; ++i) {
		equal &= a.d[i] == b.d[i];
	}
	EXPECT_FALSE(equal);
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(SizeRandom, simd128) {
	check_size_random<_uint8x16_t, _int8x16_t>();
	check_size_random<_uint16x8_t, _int16x8_t>();
	check_size_random<_uint32x4_t, _int32x4_t>();
	check_size_random<_uint64x2_t, _int64x2_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
