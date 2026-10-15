#include <cstdint>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include "compression/compression.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(bwt, simple) {
    constexpr size_t s = 1<<10;
    constexpr size_t sp = 1<<4;
	uint8_t *t1 = (uint8_t *)malloc(s);
	// NOTE: the text must end with END_MARKER, which must be its smallest symbol
	for (size_t i = 0; i < sp - 1; ++i) {
		t1[i] = 'a' + ((i*5) % 7);
	}
	t1[sp - 1] = END_MARKER;

	const int n = bwt_inplace(t1, sp);
	EXPECT_EQ(n, sp);
	uint8_t *t2 = bwt_reverse(t1, n);

    for (uint32_t i = 0; i < sp - 1; i++) {
        EXPECT_EQ(t2[i], 'a' + ((i*5) % 7));
    }
    EXPECT_EQ(t2[sp - 1], END_MARKER);

	free(t1); free(t2);
}

TEST(bwt, lcp) {
	// sorted suffixes of "banana$": $, a$, ana$, anana$, banana$, na$, nana$
	constexpr int n = 7;
	const uint8_t text[n] = {'b','a','n','a','n','a',END_MARKER};
	const uint8_t bwt[n]  = {'a','n','n','b',END_MARKER,'a','a'};
	const int lcp[n]      = {0, 0, 1, 3, 0, 0, 2};

	// exact size buffers, so ASan catches reads past the end
	uint8_t *T = (uint8_t *)malloc(n);
	int *LCP = (int *)malloc(n * sizeof(int));
	memcpy(T, text, n);
	bwt_lcp_inplace(T, n, LCP);
	for (int i = 0; i < n; i++) {
		EXPECT_EQ(bwt[i], T[i]);
		EXPECT_EQ(lcp[i], LCP[i]);
	}
	free(T); free(LCP);
}

/// naive reference: sort all suffixes
static void naive_bwt_lcp(const std::vector<uint8_t> &t, std::vector<uint8_t> &bwt, std::vector<int> &lcp) {
	const int n = int(t.size());
	std::vector<int> sa(n);
	for (int i = 0; i < n; ++i) { sa[i] = i; }
	std::sort(sa.begin(), sa.end(), [&](int a, int b) {
		return std::lexicographical_compare(t.begin() + a, t.end(), t.begin() + b, t.end());
	});
	bwt.resize(n); lcp.assign(n, 0);
	for (int i = 0; i < n; ++i) {
		bwt[i] = t[(sa[i] + n - 1) % n];
		if (i > 0) {
			int l = 0;
			while ((sa[i] + l < n) && (sa[i - 1] + l < n) && (t[sa[i] + l] == t[sa[i - 1] + l])) { ++l; }
			lcp[i] = l;
		}
	}
}

/// random texts over alphabets of 1, 2, 4 and 26 symbols, all lengths
/// 2..300: BWT, LCP and the inverse against the naive reference
TEST(bwt, random_vs_naive) {
	for (const uint32_t sigma : {1u, 2u, 4u, 26u}) {
		for (int n = 2; n <= 300; n += 7) {
			std::vector<uint8_t> t(n);
			for (int i = 0; i < n - 1; ++i) { t[i] = uint8_t('a' + rand() % sigma); }
			t[n - 1] = END_MARKER;
			std::vector<uint8_t> e_bwt; std::vector<int> e_lcp;
			naive_bwt_lcp(t, e_bwt, e_lcp);

			std::vector<uint8_t> T = t;
			ASSERT_EQ(bwt_inplace(T.data(), n), n);
			ASSERT_EQ(T, e_bwt) << sigma << " " << n;

			uint8_t *rev = bwt_reverse(T.data(), n);
			ASSERT_TRUE(std::equal(t.begin(), t.end(), rev)) << sigma << " " << n;
			free(rev);

			std::vector<uint8_t> T2 = t;
			std::vector<int> LCP(n);
			bwt_lcp_inplace(T2.data(), n, LCP.data());
			ASSERT_EQ(T2, e_bwt) << sigma << " " << n;
			ASSERT_EQ(LCP, e_lcp) << sigma << " " << n;
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
