#include <cstdint>
#include <gtest/gtest.h>

#include "combination/grsnegative.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// reference: the GRS value at x is negative iff the number of (overlapping)
/// `11` pairs in the binary representation of x is odd (OEIS A020985)
static uint64_t grs_negative_ref(const uint64_t x) {
	uint32_t c = 0;
	for (uint32_t b = 0; b < 63; ++b) { c += ((x >> b) & 3u) == 3u; }
	return c & 1u;
}

TEST(grsnegative, q) {
	// the first indices with a negative value (see `grs_negative_q`)
	constexpr uint64_t neg[] = {3, 6, 11, 12, 13, 15, 19, 22, 24, 25, 26, 30, 35, 38};
	for (const uint64_t n : neg) { EXPECT_EQ(gsr_negative::grs_negative_q(n), 1u) << n; }
	for (uint64_t x = 0; x < (1u << 16); ++x) {
		EXPECT_EQ(gsr_negative::grs_negative_q(x), grs_negative_ref(x)) << x;
	}
}

TEST(grsnegative, next) {
	uint64_t g = gsr_negative::grs_negative_q(0);
	for (uint64_t k = 0; k < (1u << 16); ++k) {
		ASSERT_EQ(g, gsr_negative::grs_negative_q(k)) << k;
		g = gsr_negative::grs_next(k, g);
	}

	// across the 32 bit boundary
	for (const uint64_t k : {0xffffffffull, 0x1ffffffffull, 0x5555555555ull, 0xfffffffffffeull}) {
		EXPECT_EQ(gsr_negative::grs_next(k, gsr_negative::grs_negative_q(k)),
		          gsr_negative::grs_negative_q(k + 1u)) << k;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
