#include <gtest/gtest.h>
#include <cstdint>

#include "container/binary_packed_vector.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// get_bits(i, j) and higher_mask against a bit by bit reference, for every
/// window of at most one limb (same limb and across two limbs)
template<const uint32_t n, typename T>
static void check_get_bits() {
	using V = FqPackedVector<n, 2, T>;
	constexpr uint32_t bits = sizeof(T) * 8;
	V v;
	v.random();
	for (uint32_t i = 0; i < n; ++i) {
		const T hm = V::higher_mask(i);
		for (uint32_t b = 0; b < bits; ++b) {
			ASSERT_EQ((hm >> b) & 1u, T(b >= (i % bits))) << i << " " << b;
		}
		for (uint32_t j = i + 1; (j <= n) && (j - i <= bits); ++j) {
			T e = 0;
			for (uint32_t k = i; k < j; ++k) { e |= T(v.get_bit_shifted(k)) << (k - i); }
			ASSERT_EQ(v.get_bits(i, j), e) << i << " " << j;
		}
	}
}

TEST(BinaryPackedVector, get_bits_higher_mask) {
	for (uint32_t r = 0; r < 20; ++r) {
		check_get_bits<200, uint64_t>();
		check_get_bits<100, uint32_t>();
		check_get_bits<64, uint64_t>();
		check_get_bits<37, uint8_t>();
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
