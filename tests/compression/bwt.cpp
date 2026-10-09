#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
