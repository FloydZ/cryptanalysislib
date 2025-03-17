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
	for (size_t i = 0; i < sp; ++i) {
		t1[i] = i+i;// (i*5)/7;
	}

	const int n = bwt_inplace(t1, sp);
	uint8_t *t2 = bwt_reverse(t1, n);

    for (uint32_t i = 0; i < sp; i++) {
        EXPECT_EQ(t2[i], i+i);
    }

	free(t1); free(t2);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
