#include <cstddef>
#include <bitset>
#include <set>
#include <gtest/gtest.h>

#include "combination/revolving_door.h"
#include "math/math.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(revolvingDoor, simple) {
    combination_revdoor c(10, 2);
    uint32_t k1 = 0, k2 = 1;
    do {
    	c.print_deltaset();
    	std::cout << " " << k1 << " " << k2 << std::endl;
    } while (c.next(&k1, &k2));
    return;
}

/// all k-subsets exactly once, every step removes the element `k1` and adds
/// `k2` (the new combination is checked against the reported change)
TEST(revolvingDoor, all_combinations) {
	for (uint32_t n = 1; n <= 12; ++n) {
		for (uint32_t k = 1; k <= n; ++k) {
			combination_revdoor c(n, k);
			auto mask = [&]() { uint64_t m = 0; for (uint32_t i = 0; i < k; ++i) { m |= 1ull << c.data()[i]; } return m; };
			std::set<uint64_t> seen{mask()};
			uint64_t prev = mask();
			uint32_t k1 = 0, k2 = 0;
			while (c.next(&k1, &k2)) {
				const uint64_t m = mask();
				EXPECT_EQ(__builtin_popcountll(m), k) << n << " " << k;
				EXPECT_EQ(prev ^ m, (1ull << k1) ^ (1ull << k2)) << n << " " << k;
				EXPECT_TRUE((prev >> k1) & 1u);
				EXPECT_TRUE((m >> k2) & 1u);
				seen.insert(m);
				prev = m;
			}
			EXPECT_EQ(seen.size(), bc(n, k)) << n << " " << k;
		}
	}
}

/// `next` is constexpr now (no gotos)
constexpr uint32_t count_constexpr(const uint32_t n, const uint32_t k) {
	combination_revdoor c(n, k);
	uint32_t k1 = 0, k2 = 0, ctr = 1;
	while (c.next(&k1, &k2)) { ctr += 1; }
	return ctr;
}
static_assert(count_constexpr(8, 3) == 56);
static_assert(count_constexpr(10, 4) == 210);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
