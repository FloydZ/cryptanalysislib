#include <cstdint>
#include <gtest/gtest.h>
#include <cstring>
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

/// compresses and decompresses `in`, checks that nothing changed
/// \return compressed size
static size_t roundtrip(const uint8_t *in, const size_t s) {
	uint8_t *t1 = (uint8_t *)malloc(s + 1);
	uint8_t *t2 = (uint8_t *)malloc(CompressBound(s));
	uint8_t *t3 = (uint8_t *)malloc(s + 1);
	memcpy(t1, in, s);

	const size_t newsize = CompressData(t1, t2, s, MAX_WINDOWSIZE, nullptr);
	const size_t decompressed_size = DecompressData(t2, t3);

	EXPECT_LE(newsize, CompressBound(s));
	EXPECT_EQ(decompressed_size, s);
	for (size_t i = 0; i < s; ++i) {
		EXPECT_EQ(t3[i], t1[i]);
	}

	free(t1);free(t2);free(t3);
	return newsize;
}

TEST(lzss, random) {
	// random data is not compressible, but must survive the roundtrip
	constexpr size_t s = 1<<15;
	uint8_t *t = (uint8_t *)malloc(s);
	for (size_t i = 0; i < s; ++i) {
		t[i] = rng();
	}
	roundtrip(t, s);
	free(t);
}

TEST(lzss, repetitive) {
	constexpr size_t s = 1<<15;
	uint8_t *t = (uint8_t *)malloc(s);
	for (size_t i = 0; i < s; ++i) {
		t[i] = (i % 37) < 20 ? 'a' + (i % 7) : rng() % 4;
	}
	EXPECT_LT(roundtrip(t, s), s / 2);
	free(t);
}

TEST(lzss, small) {
	// includes lengths where the end marker needs a new control byte
	uint8_t t[64];
	for (size_t i = 0; i < 64; ++i) {
		t[i] = rng() % 3;
	}
	for (size_t s = 0; s <= 64; ++s) {
		roundtrip(t, s);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
