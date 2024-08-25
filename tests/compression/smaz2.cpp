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

TEST(smaz2, simple) {
	const unsigned char text[] = "this is an input text, test, test, test";
	unsigned char text2[sizeof(text)] = {0};
	unsigned char buf[256];

	const size_t olen  = smaz2_compress(buf, sizeof(buf), text, sizeof(text));
	printf("Compressed length (%lu): %.02f%%\n", olen, (float)olen/sizeof(text)*100);
	const size_t olen2 = smaz2_decompress(text2, sizeof(text2), buf, olen);
	EXPECT_EQ(olen2, sizeof(text));
	EXPECT_EQ(memcmp(text, text2, sizeof(text)), 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
