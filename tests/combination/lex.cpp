#include <cstdint>
#include <set>
#include <gtest/gtest.h>

#include "combination/lex.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using L = enumeration_lexicographic;

/// the subset-lexrev order of all 2**n words of length n, see `next_lexrev`
TEST(lex, next_lexrev_n4) {
	// the sequence of the comment of `next_lexrev`, MSB first
	constexpr uint64_t e[16] = {0b1000, 0b1100, 0b1110, 0b1111, 0b1101, 0b1010, 0b1011, 0b1001,
	                            0b0100, 0b0110, 0b0111, 0b0101, 0b0010, 0b0011, 0b0001, 0b0000};
	uint64_t x = e[0];
	for (uint32_t i = 1; i < 16; ++i) {
		x = L::next_lexrev(x);
		EXPECT_EQ(x, e[i]) << i;
	}
}

TEST(lex, next_lexrev_all_words) {
	for (uint32_t n = 1; n <= 12; ++n) {
		std::set<uint64_t> seen;
		uint64_t x = 1ull << (n - 1u);
		for (uint64_t i = 0; i < (1ull << n); ++i) {
			EXPECT_LT(x, 1ull << n);
			seen.insert(x);
			x = L::next_lexrev(x);
		}
		EXPECT_EQ(seen.size(), 1ull << n);
	}
}

TEST(lex, negidx_roundtrip) {
	for (uint64_t k = 0; k < 10000; ++k) {
		EXPECT_EQ(L::lexrev2negidx(L::negidx2lexrev(k)), k);
	}
}

TEST(lex, fixed_points) {
	EXPECT_TRUE(L::is_lexrev_fixed_point(0));
	EXPECT_TRUE(L::is_lexrev_fixed_point(1));
	EXPECT_FALSE(L::is_lexrev_fixed_point(3));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
