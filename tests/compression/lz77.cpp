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

using Label_Type = uint64_t;

TEST(lz77, simple) {
	const unsigned char text[] = "this aaaaaaaaaaaaaaaaaa is an 283682 ithis is an 283682 input text, alsdkfj test, aaaaaaaaaaaaaaaaaaa nput text, alsdkfj test,  lhansdf 8273e 8273e test, test";
	unsigned char text2[sizeof(text)] = {0};
	unsigned char buf[1024];

	for (int i = 1; i < 16; ++i) {
		const size_t olen  = lz77_compress(buf, text, sizeof(text), i);
		printf("Compressed length %u: (%lu): %.02f%%\n", i, olen, (float)olen/sizeof(text)*100);
		const size_t olen2 = lz77_decompress(text2, buf);
		EXPECT_EQ(olen2, sizeof(text));
		EXPECT_EQ(memcmp(text, text2, sizeof(text)), 0);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
