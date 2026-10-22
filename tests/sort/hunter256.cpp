#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

// up to 256 inputs: `SortWord_t` is `SortWord256_t`
#define NMAX 256
#include "sort/sorting_network/hunter.h"

using ::testing::InitGoogleTest;

static_assert(std::is_same_v<SortWord_t, SortWord256_t>);

/// insertion sort for `n` inputs
static Network_t insertion(const u8 n) {
	Network_t nw;
	for (u32 i = 1; i < n; ++i) {
		for (u32 j = i; j > 0; --j) { nw.push_back({(u8)(j - 1), (u8)j}); }
	}
	return nw;
}

TEST(hunter256, word) {
	const SortWord_t one = 1, zero = 0, all = ~SortWord_t(0);
	for (uint32_t s = 0; s < 256; ++s) {
		const SortWord_t x = one << s;
		EXPECT_EQ(x.w[s / 64], 1ull << (s % 64)) << s;
		EXPECT_EQ(x >> s, one) << s;
		EXPECT_TRUE(x > zero) << s;
		if (s > 0) { EXPECT_TRUE((one << (s - 1)) < x) << s; }
		// the carry through all lower limbs: (x - 1) + 1
		EXPECT_EQ(~(all << s) + one, x) << s;
	}

	const SortWord_t a = 0x0123456789abcdefull;
	for (uint32_t s = 0; s < 192; ++s) { EXPECT_EQ((a << s) >> s, a) << s; }
	// the highest 8 bits of `a` are shifted out
	EXPECT_EQ((a << 200) >> 200, a & ~(all << 56));
	EXPECT_EQ(all + one, zero);

	SortWord_t p = 3;
	p *= 1ull << 63;
	EXPECT_EQ(p.w[0], 1ull << 63);
	EXPECT_EQ(p.w[1], 1u);
	p = one << 255;
	p *= 2;
	EXPECT_EQ(p, zero);
}

TEST(hunter256, layers_depth) {
	EXPECT_EQ(computeDepth(Network_t{{0, 100}, {150, 200}, {64, 254}}), 1u);
	EXPECT_EQ(computeDepth(Network_t{{0, 100}, {100, 200}, {0, 200}}), 3u);
	EXPECT_EQ(computeLayers(Network_t{{0, 100}, {150, 200}, {64, 254}}).size(), 1u);
	EXPECT_TRUE(isSymmetric(250, Network_t{{0, 249}, {100, 149}}));
	EXPECT_FALSE(isSymmetric(250, Network_t{{0, 249}, {100, 148}}));
}

/// the 0/1 outputs of a sorting network on 200 lines are the 201 sorted ones
TEST(hunter256, prefix_outputs) {
	constexpr u8 n = 200;
	Network_t nw = insertion(n);
	SinglePatternList_t p;
	computePrefixOutputs(n, nw, p);
	ASSERT_EQ(p.size(), n + 1u);
	std::sort(p.begin(), p.end());
	for (u32 k = 0; k <= n; ++k) {
		// the k highest lines are 1
		const SortWord_t e = (k == 0) ? SortWord_t(0) : ((~SortWord_t(0) >> (256 - k)) << (n - k));
		EXPECT_EQ(p[k], e) << k;
	}

	// without the last comparator (0, 1) the network does not sort
	nw.pop_back();
	computePrefixOutputs(n, nw, p);
	EXPECT_EQ(p.size(), n + 2u);
}

/// a single comparator on lines >= 64 applied to a sorted pattern list
TEST(hunter256, swap_sortedpatterns) {
	const u8 lines[4] = {70, 130, 200, 250};
	SinglePatternList_t all;
	for (uint32_t i = 0; i < 16; ++i) {
		SortWord_t w = 0;
		for (uint32_t b = 0; b < 4; ++b) {
			if ((i >> b) & 1u) { w |= SortWord_t(1) << lines[b]; }
		}
		all.push_back(w);
	}
	std::sort(all.begin(), all.end());

	for (u32 a = 0; a < 4; ++a) {
		for (u32 b = a + 1; b < 4; ++b) {
			const SortWord_t lo = SortWord_t(1) << lines[a], hi = SortWord_t(1) << lines[b];
			SinglePatternList_t e;
			for (const SortWord_t &w : all) {
				e.push_back((((w & lo) == lo) && ((w & hi) == 0)) ? (w ^ lo ^ hi) : w);
			}
			std::sort(e.begin(), e.end());
			e.erase(std::unique(e.begin(), e.end()), e.end());

			SinglePatternList_t r = all;
			swap_sortedpatterns(r, Pair_t{lines[a], lines[b]});
			EXPECT_EQ(r, e) << a << " " << b;
		}
	}
}

/// every greedy pair joins two free lines: 3 of their 4 outputs are left
TEST(hunter256, greedy_prefix) {
	Network_t prefix;
	const SortWord_t size = createGreedyPrefix(130, 8, false, prefix);
	EXPECT_EQ(prefix.size(), 8u);
	EXPECT_EQ(size, SortWord_t(6561) << (130 - 16));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
