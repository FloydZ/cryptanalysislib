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
	// const unsigned char text[] = "this aaaaaaaaaaaaaaaaaa is an 283682 ithis is an 283682 aaaaaaaaaaaaaaa input text, alsdkfj test, aaaaaaaaaaaaaaaaaaa nput text, alsdkfj test,  lhansdf 8273e 8273e test, test aaaaaaaaaaaaaaa";
	// unsigned char text2[sizeof(text)] = {0};
	const uint8_t  text[] = {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23,24,24,25,25,26,26,27,27,28,28,29};
	uint8_t text2[sizeof(text)];
	unsigned char buf[1024];

	for (int i = 1; i < 16; ++i) {
		const size_t olen  = lz77_compress(buf, (uint8_t *)text, sizeof(text), i);
		printf("Compressed length %u: (%lu): %.02f%%\n", i, olen, (float)olen/sizeof(text)*100);
		const size_t olen2 = lz77_decompress((uint8_t *)text2, buf);
		EXPECT_EQ(olen2, sizeof(text));
		EXPECT_EQ(memcmp(text, text2, sizeof(text)), 0);
	}
}

TEST(lz77, repeated_tail) {
	// matches running into the end of the input; exact size buffers so
	// ASan catches reads past the input
	for (size_t n = 1; n < 64; n++) {
		for (uint8_t w = 1; w < 16; w += 2) {
			uint8_t *text = (uint8_t *)malloc(n);
			memset(text, 'a', n);
			text[0] = 'b';
			uint8_t *buf = (uint8_t *)malloc(5 + 3 * n);
			uint8_t *text2 = (uint8_t *)malloc(n);

			const size_t olen = lz77_compress(buf, text, n, w);
			EXPECT_LE(olen, 5 + 3 * n);
			EXPECT_EQ(n, lz77_decompress(text2, buf));
			EXPECT_EQ(0, memcmp(text, text2, n));
			free(text); free(buf); free(text2);
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
