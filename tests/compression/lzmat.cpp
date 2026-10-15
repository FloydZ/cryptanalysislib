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
		t1[i] = i+i;// (i*5)/7;
	}

	uint32_t out_size=MAX_LZMAT_ENCODED_SIZE(s), in_size=s;
	auto f = lzmat_encode(t2, &out_size, t1, s);
	EXPECT_EQ(f, 0);
	lzmat_decode(t3, &in_size, t2, out_size);
	printf("Compressed length: (%u): %.02f%%\n", out_size, (float)out_size/sizeof(s)*100);
	for (size_t i = 0; i < s; ++i) {
		EXPECT_EQ(t3[i], t1[i]);
	}

	free(t1);free(t2);free(t3);
}

TEST(lzmat, all_sizes) {
	// exact size buffers, so ASan catches reads past the input. Sizes 8k+1
	// of incompressible data used to fail with `MAX_LZMAT_ENCODED_SIZE`.
	for (uint32_t s = 1; s < 300; ++s) {
		uint8_t *t1 = (uint8_t *)malloc(s);
		uint8_t *t2 = (uint8_t *)malloc(MAX_LZMAT_ENCODED_SIZE(s));
		uint8_t *t3 = (uint8_t *)malloc(s);
		for (size_t i = 0; i < s; ++i) {
			t1[i] = (s & 1) ? i + i : 'a' + (i % 3);
		}

		uint32_t out_size = MAX_LZMAT_ENCODED_SIZE(s), in_size = s;
		EXPECT_EQ(0, lzmat_encode(t2, &out_size, t1, s));
		EXPECT_EQ(0, lzmat_decode(t3, &in_size, t2, out_size));
		EXPECT_EQ(s, in_size);
		EXPECT_EQ(0, memcmp(t1, t3, s));
		free(t1); free(t2); free(t3);
	}
}

TEST(lzmat, empty) {
	uint8_t in[1] = {0}, out[0x40], dec[1];
	uint32_t out_size = sizeof(out), dec_size = sizeof(dec);
	EXPECT_EQ(0, lzmat_encode(out, &out_size, in, 0));
	EXPECT_EQ(0u, out_size);
	EXPECT_EQ(0, lzmat_decode(dec, &dec_size, out, 0));
	EXPECT_EQ(0u, dec_size);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
