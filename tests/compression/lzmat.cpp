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
    
    uint32_t out_size, in_size;
	lzmat_encode(t2, &out_size, t1, s, nullptr);
	lzmat_decode(t3, &in_size, t2, out_size);
	printf("Compressed length: (%lu): %.02f%%\n", out_size, (float)out_size/sizeof(s)*100);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
