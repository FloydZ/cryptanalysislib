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

TEST(lzmat, simple) {
    constexpr size_t s = 1<<10;
	uint8_t *t1 = (uint8_t *)malloc(s);
	uint8_t *t2 = (uint8_t *)malloc(s);
	uint8_t *t3 = (uint8_t *)malloc(s);
	for (size_t i = 0; i < s; ++i) {
		t1[i] = 1;
	}

	uint32_t out_size=MAX_LZMAT_ENCODED_SIZE(s), in_size=s;
	auto f = lzmat_encode(t2, &out_size, t1, s);
	ASSERT_EQ(f, 0);
	lzmat_decode(t3, &in_size, t2, out_size);
	printf("Compressed length: (%u): %.02f%%\n", out_size, (float)out_size/sizeof(s)*100);
	for (size_t i = 0; i < s; ++i) {
		ASSERT_EQ(t3[i], t1[i]);
	}

	free(t1);free(t2);free(t3);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
