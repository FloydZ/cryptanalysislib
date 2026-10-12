#include <gtest/gtest.h>
#include <cstdint>
#include <set>

#include "combination/min_change.h"
#include "math/math.h"

using ::testing::InitGoogleTest;

/// all k-subsets of n bits exactly once, two neighbours differ by one
/// cleared and one set bit
TEST(min_change, all_combinations) {
	for (uint32_t n = 1; n <= 14; ++n) {
		for (uint32_t k = 0; k <= n; ++k) {
			enumeration_min_change<uint64_t> e(n, k);
			std::set<uint64_t> seen;
			uint64_t prev = e.current();
			seen.insert(prev);
			while (e.next()) {
				const uint64_t c = e.current();
				EXPECT_EQ(__builtin_popcountll(c), k) << n << " " << k;
				EXPECT_LT(c, 1ull << n) << n << " " << k;
				EXPECT_EQ(__builtin_popcountll(c ^ prev), 2) << n << " " << k;
				seen.insert(c);
				prev = c;
				ASSERT_LE(seen.size(), bc(n, k)) << "too many " << n << " " << k;
			}
			EXPECT_EQ(seen.size(), bc(n, k)) << n << " " << k;
			EXPECT_FALSE(e.next());
		}
	}
}

/// the example from the docs (k = 3, n = 5) and the static helpers
TEST(min_change, example) {
	using E = enumeration_min_change<uint32_t>;
	const uint32_t expect[] = {0b00111, 0b01101, 0b01110, 0b01011, 0b11001,
	                           0b11010, 0b11100, 0b10101, 0b10110, 0b10011};
	E e(5, 3);
	for (const uint32_t x : expect) {
		EXPECT_EQ(e.current(), x);
		e.next();
	}
	static_assert(E::gray_code(E::inverse_gray_code(0x12345678ull)) == 0x12345678ull);
	static_assert(E::is_pow_of_2(0) && E::is_pow_of_2(8) && !E::is_pow_of_2(6));
	static_assert(E::next_minchange_comb(0b111, E::igc_last_comb(3, 5)) == 0b01101);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
