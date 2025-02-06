#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "compression/compression.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(lzmat, simple) {
    constexpr size_t s = 1<<15;
	uint8_t *t1 = (uint8_t *)malloc(s);
	uint8_t *t2 = (uint8_t *)malloc(s);
	uint8_t *t3 = (uint8_t *)malloc(s);
	for (size_t i = 0; i < s; ++i) {
		t1[i] = i + s;
	}

    const size_t newsize = CompressData(t1, t2, s, MAX_WINDOWSIZE, &CompressCallback);
	const size_t decompressed_size = DecompressData(t2, t3);
	for (size_t i = 0; i < s; ++i) {
		EXPECT_EQ(t3[i], t1[i]);
	}

    EXPECT_EQ(decompressed_size, s);
	EXPECT_LE(newsize, s);
	free(t1);free(t2);free(t3);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}