#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "sort/sorting_network/hunter.h"

using ::testing::InitGoogleTest;

/// optimal 4 input network: 5 pairs, depth 3
static const Network_t nw4 = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}};

/// Batcher's odd-even merge sort for 8 inputs: 19 pairs, depth 6
static const Network_t nw8 = {
	{0, 1}, {2, 3}, {4, 5}, {6, 7},
	{0, 2}, {1, 3}, {4, 6}, {5, 7},
	{1, 2}, {5, 6},
	{0, 4}, {1, 5}, {2, 6}, {3, 7},
	{2, 4}, {3, 5},
	{1, 2}, {3, 4}, {5, 6}};

TEST(hunter, layers_depth) {
	const auto l = computeLayers(nw4);
	ASSERT_EQ(l.size(), 3u);
	EXPECT_EQ(l[0], (Network_t{{0, 1}, {2, 3}}));
	EXPECT_EQ(l[1], (Network_t{{0, 2}, {1, 3}}));
	EXPECT_EQ(l[2], (Network_t{{1, 2}}));
	EXPECT_EQ(computeDepth(nw4), 3u);
	EXPECT_EQ(computeDepth(nw8), 6u);
	EXPECT_EQ(computeLayers(nw8).size(), 6u);
	EXPECT_EQ(computeLayers(Network_t{}).size(), 0u);
}

TEST(hunter, is_sorting_network) {
	EXPECT_TRUE(isSortingNetwork(4, nw4));
	EXPECT_TRUE(isSortingNetwork(8, nw8));
	// every proper prefix of the networks does not sort
	for (size_t k = 0; k < nw8.size(); ++k) {
		const Network_t p(nw8.begin(), nw8.begin() + k);
		EXPECT_FALSE(isSortingNetwork(8, p)) << k;
	}
	// bubble sort for 10 inputs
	Network_t bubble;
	for (u8 i = 0; i < 10; ++i) {
		for (u8 j = 0; j + 1 < 10 - i; ++j) { bubble.push_back({j, (u8)(j + 1)}); }
	}
	EXPECT_TRUE(isSortingNetwork(10, bubble));
	EXPECT_TRUE(isSortingNetwork(1, Network_t{}));
}

TEST(hunter, is_symmetric) {
	EXPECT_TRUE(isSymmetric(4, nw4));
	EXPECT_TRUE(isSymmetric(8, nw8));
	// (0,1) and its mirror (1,2) are in different layers
	EXPECT_FALSE(isSymmetric(3, Network_t{{0, 1}, {1, 2}, {0, 1}}));
	EXPECT_FALSE(isSymmetric(4, Network_t{{0, 1}, {0, 2}}));
	EXPECT_TRUE(isSymmetric(4, Network_t{{1, 2}}));
}

TEST(hunter, simd_swap_pattern) {
	EXPECT_EQ((Pair_t{2, 7}.size()), 5u);
	EXPECT_TRUE(isSimdSwapPattern(Network_t{}));
	EXPECT_TRUE(isSimdSwapPattern(Network_t{{0, 1}, {2, 3}, {4, 5}}));
	EXPECT_TRUE(isSimdSwapPattern(Network_t{{0, 2}, {1, 3}}));
	EXPECT_FALSE(isSimdSwapPattern(Network_t{{0, 3}, {1, 2}}));
	// every layer of the 4 input network is one
	for (const auto &l : computeLayers(nw4)) {
		EXPECT_TRUE(isSimdSwapPattern(l));
	}
	// the earliest layer packing mixes (1,2) and (0,4) in Batcher's network
	EXPECT_FALSE(isSimdSwapPattern(computeLayers(nw8)[2]));
}

TEST(hunter, symmetric_expansion) {
	Network_t out;
	symmetricExpansion(4, Network_t{{0, 1}, {1, 2}, {0, 2}}, out);
	EXPECT_EQ(out, (Network_t{{0, 1}, {2, 3}, {1, 2}, {0, 2}, {1, 3}}));
	EXPECT_TRUE(isSymmetric(4, out));
}

TEST(hunter, prefix_outputs) {
	SinglePatternList_t p;
	computePrefixOutputs(4, Network_t{}, p);
	EXPECT_EQ(p.size(), 16u);
	computePrefixOutputs(4, nw4, p);
	EXPECT_EQ(p.size(), 5u);
	computePrefixOutputs(8, nw8, p);
	EXPECT_EQ(p.size(), 9u);
	computePrefixOutputs(8, Network_t(nw8.begin(), nw8.begin() + 4), p);
	EXPECT_EQ(p.size(), 81u);
}

/// a single comparator applied to a sorted pattern list against brute force
TEST(hunter, swap_sortedpatterns) {
	for (u8 lo = 0; lo < 5; ++lo) {
		for (u8 hi = lo + 1; hi < 6; ++hi) {
			SinglePatternList_t all(64);
			for (uint32_t i = 0; i < 64; ++i) { all[i] = i; }
			std::vector<SortWord_t> e;
			for (const SortWord_t w : all) {
				SortWord_t x = w;
				if (((x >> lo) & 1u) && !((x >> hi) & 1u)) {
					x ^= (1ull << lo) | (1ull << hi);
				}
				e.push_back(x);
			}
			std::sort(e.begin(), e.end());
			e.erase(std::unique(e.begin(), e.end()), e.end());
			swap_sortedpatterns(all, Pair_t{lo, hi});
			EXPECT_EQ(all, e);
		}
	}
}

TEST(hunter, greedy_prefix) {
	for (const bool sym : {false, true}) {
		Network_t prefix;
		const SortWord_t size = createGreedyPrefix(8, 10, sym, prefix);
		SinglePatternList_t p;
		computePrefixOutputs(8, prefix, p);
		EXPECT_EQ(size, p.size());
		EXPECT_LT(size, 256u);
		if (sym) { EXPECT_TRUE(isSymmetric(8, prefix)); }
	}
}

TEST(hunter, och) {
	OCH_t o;
	EXPECT_TRUE(o.improved(19, 6));
	EXPECT_FALSE(o.improved(19, 6));
	EXPECT_FALSE(o.improved(20, 7));
	EXPECT_TRUE(o.improved(20, 5));
	EXPECT_TRUE(o.improved(18, 7));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
