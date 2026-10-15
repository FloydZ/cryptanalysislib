#include <gtest/gtest.h>
#include <cstdint>

#include "container/binary_packed_vector.h"
#include "random.h"

using ::testing::InitGoogleTest;

/// rol/ror against a bitwise reference, and rol(ror(x)) == x
template<const uint32_t n, typename T>
static void check() {
	using V = FqPackedVector<n, 2, T>;
	V a, b, c;
	a.random();
	for (uint32_t s = 0; s < n; ++s) {
		V::rol(b, a, s);
		for (uint32_t j = 0; j < n; ++j) {
			ASSERT_EQ(b.get_bit_shifted((j + s) % n), a.get_bit_shifted(j)) << s << " " << j;
		}
		V::ror(c, b, s);
		EXPECT_TRUE(c.is_equal(a)) << s;
		V::ror(b, a, s);
		for (uint32_t j = 0; j < n; ++j) {
			ASSERT_EQ(b.get_bit_shifted(j), a.get_bit_shifted((j + s) % n)) << s << " " << j;
		}
	}
}

TEST(BinaryPackedVector, rotate) {
	check<1, uint64_t>();
	check<64, uint64_t>();
	check<100, uint64_t>();
	check<257, uint64_t>();
	check<77, uint32_t>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	cryptanalysislib::rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
