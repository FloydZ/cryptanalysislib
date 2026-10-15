#include <cstdint>
#include <set>
#include <gtest/gtest.h>

#include "combination/subset_gray.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// all subsets of `v` are visited once and consecutive subsets differ in one bit
TEST(subset_gray, next) {
	for (const uint64_t v : {0x1ull, 0xbull, 0xf0ull, 0x1234ull, 0x8000000000000005ull}) {
		bit_subset_gray_T<uint64_t> G(v);
		std::set<uint64_t> seen;
		uint64_t p = G.first();
		EXPECT_EQ(p, 0u);
		seen.insert(p);
		const uint64_t N = 1ull << __builtin_popcountll(v);
		for (uint64_t i = 1; i < N; ++i) {
			EXPECT_TRUE(G.next());
			const uint64_t q = G.current();
			EXPECT_EQ(__builtin_popcountll(p ^ q), 1);
			EXPECT_EQ(q & ~v, 0u);
			seen.insert(q);
			p = q;
		}
		EXPECT_EQ(seen.size(), N);
		EXPECT_FALSE(G.next());
		EXPECT_EQ(G.current(), p);
		EXPECT_EQ(G.full_set(), v);
	}
}

TEST(subset_gray, last_prev) {
	const uint64_t v = 0b1011010;
	bit_subset_gray_T<uint64_t> G(v);
	// the highest one of the full set
	EXPECT_EQ(G.last(), 0b1000000u);
	std::set<uint64_t> seen;
	uint64_t p = G.current();
	seen.insert(p);
	for (uint64_t i = 1; i < (1u << 4); ++i) {
		EXPECT_TRUE(G.previous());
		const uint64_t q = G.current();
		EXPECT_EQ(__builtin_popcountll(p ^ q), 1);
		seen.insert(q);
		p = q;
	}
	EXPECT_EQ(seen.size(), 1u << 4);
	EXPECT_FALSE(G.previous());
	EXPECT_EQ(G.current(), 0u);

	EXPECT_EQ(G.first(0b11), 0u);
	EXPECT_EQ(G.full_set(), 0b11u);
	EXPECT_EQ(G.last(0b110), 0b100u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
