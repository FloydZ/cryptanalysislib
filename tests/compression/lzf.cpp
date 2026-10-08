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

TEST(lzf, simple) {
	const uint8_t  text[] = {128,11,12,10,10,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29};
	uint8_t text2[sizeof(text)];
	unsigned char buf[1024];

	const uint32_t new_size = lzf_compress((uint8_t *)text, sizeof(text), buf, 1024);
	printf("Compressed length %u: %.02f%%\n", new_size, ((float)sizeof(text)/(float)new_size)*100);
	const size_t olen2 = lzf_decompress(buf, new_size, text2, sizeof(text));
	EXPECT_EQ(olen2, sizeof(text));
	EXPECT_EQ(memcmp(text, text2, sizeof(text)), 0);
}

TEST(lzf, small_sizes) {
	// exact size inputs, so ASan catches reads past the input
	for (uint32_t n = 1; n < 40; n++) {
		uint8_t *text = (uint8_t *)malloc(n);
		for (uint32_t i = 0; i < n; i++) { text[i] = (i & 4) ? 'a' : uint8_t(i); }
		unsigned char buf[128];
		uint8_t text2[40];

		const uint32_t new_size = lzf_compress(text, n, buf, sizeof(buf));
		EXPECT_GT(new_size, 0u);
		EXPECT_EQ(n, lzf_decompress(buf, new_size, text2, n));
		EXPECT_EQ(0, memcmp(text, text2, n));
		free(text);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
