#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "algorithm/transpose.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(transform, b8x8) {
	const uint64_t t0 = transpose_b8x8(0b01);
	EXPECT_EQ(t0, 1);
	const uint64_t t1 = transpose_b8x8(0b10);
	EXPECT_EQ(t1, 256);
	const uint64_t t2 = transpose_b8x8(0b100);
	EXPECT_EQ(t2, 1u<<16);
}

TEST(transform, b8x8_be) {
	const uint64_t t0 = transpose_b8x8_be(0b01);
	EXPECT_EQ(t0, 1ull<<63);
	const uint64_t t1 = transpose_b8x8_be(0b10000000);
	EXPECT_EQ(t1, 1ull<<7);
	const uint64_t t2 = transpose_b8x8_be(0b100);
	EXPECT_EQ(t2, 1ull<<47);
}

TEST(transform, b64x64) {
	uint64_t in[64] = {0};
	uint64_t out[64] = {0};
	transpose_b64x64(out, in);
	for (uint32_t i = 0; i < 64; i++) {
		EXPECT_EQ(out[i], 0);
	}

	transpose_b64x64_inplace(in);
	for (uint32_t i = 0; i < 64; i++) {
		EXPECT_EQ(out[i], 0);
	}

	for (uint32_t i = 0; i < 64; i++) {
		in[i] = cryptanalysislib::rng();
	}

	transpose_b64x64(out, in);
	transpose_b64x64_inplace(in);
	for (uint32_t i = 0; i < 64; i++) {
		EXPECT_EQ(out[i], in[i]);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
