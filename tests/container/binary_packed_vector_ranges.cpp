#include <gtest/gtest.h>
#include <cstdint>

#include "container/binary_packed_vector.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// add (xor) and mul (and) on the bit range [l, h) against a bitwise
/// reference, the bits outside of the range must be untouched. The vectors
/// are long enough that the SIMD loop runs.
template<const uint32_t n, typename T>
static void check_ranges() {
	using V = FqPackedVector<n, 2, T>;
	for (uint32_t r = 0; r < 300; ++r) {
		V a, b, c, d;
		a.random(); b.random(); c.random();
		d = c;
		const uint32_t l = rng() % (n - 1);
		const uint32_t h = l + 1 + rng() % (n - l);

		V::add(c, a, b, l, h);
		for (uint32_t k = 0; k < n; ++k) {
			const bool e = (k >= l && k < h) ? (a.get_bit_shifted(k) ^ b.get_bit_shifted(k)) : d.get_bit_shifted(k);
			ASSERT_EQ(c.get_bit_shifted(k), e) << "add " << l << " " << h << " " << k;
		}

		c = d;
		V::mul(c, a, b, l, h);
		for (uint32_t k = 0; k < n; ++k) {
			const bool e = (k >= l && k < h) ? (a.get_bit_shifted(k) & b.get_bit_shifted(k)) : d.get_bit_shifted(k);
			ASSERT_EQ(c.get_bit_shifted(k), e) << "mul " << l << " " << h << " " << k;
		}
	}
}

TEST(BinaryPackedVector, ranges) {
	check_ranges<2048, uint64_t>();
	check_ranges<1000, uint64_t>();
	check_ranges<777, uint32_t>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
