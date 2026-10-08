#include <cstdint>
#include <gtest/gtest.h>
#include <set>

#include "math/math.h"
#include "combination/necklace.h"

using ::testing::InitGoogleTest;

// OEIS A000031 (binary necklaces) and A001037 (binary Lyndon words)
static const uint64_t NECK[] = {1,2,3,4,6,8,14,20,36,60,108,188,352,632,1182,2192,4116};
static const uint64_t LYN[]  = {1,2,1,2,3,6,9,18,30,56,99,186,335,630,1161,2182,4080};

/// one word per rotation class, all within `n` bits
template<typename T, const uint32_t n>
static void check() {
	auto rotmin = [](uint64_t a) {
		const uint64_t mask = (n == 64) ? -1ull : ((1ull << n) - 1ull);
		uint64_t m = a;
		for (uint32_t r = 1; r < n; r++) {
			a = ((a << 1u) | (a >> (n - 1u))) & mask;
			m = a < m ? a : m;
		}
		return m;
	};

	bit_necklace<T, n> g;
	g.init();
	std::set<uint64_t> classes{rotmin(g.data())};
	uint64_t cnt = 1, lyn = (n == 1);
	while (g.next()) {
		ASSERT_LT(cnt, NECK[n]);
		EXPECT_EQ((uint64_t) g.data() >> (n - 1u) >> 1u, 0u);
		classes.insert(rotmin(g.data()));
		lyn += g.is_lyndon_word() ? 1 : 0;
		cnt++;
	}

	EXPECT_EQ(cnt, NECK[n]);
	EXPECT_EQ(classes.size(), NECK[n]);
	EXPECT_EQ(lyn, LYN[n]);
}

TEST(necklace, uint8) {
	check<uint8_t, 1>(); check<uint8_t, 5>(); check<uint8_t, 7>(); check<uint8_t, 8>();
}

TEST(necklace, uint16) {
	check<uint16_t, 6>(); check<uint16_t, 13>(); check<uint16_t, 16>();
}

TEST(necklace, uint32) {
	check<uint32_t, 5>(); check<uint32_t, 12>(); check<uint32_t, 16>();
}

TEST(necklace, uint64) {
	check<uint64_t, 5>(); check<uint64_t, 12>(); check<uint64_t, 16>();
}

TEST(necklace, next_lyn) {
	bit_necklace<uint64_t, 6> g;
	g.init();
	uint64_t k = 0;
	while (g.next_lyn()) {
		ASSERT_LT(k, LYN[6]);
		EXPECT_TRUE(g.is_lyndon_word());
		k++;
	}
	EXPECT_EQ(k, LYN[6]);
}

TEST(necklace, n64) {
	bit_necklace<uint64_t, 64> g;
	g.init();
	// divisors 1, 2, 4, 8, 16, 32, 64
	EXPECT_EQ(g.tfb_, 0x800000008000808bull);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
