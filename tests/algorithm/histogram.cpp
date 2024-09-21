#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/histogram.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
using namespace cryptanalysislib;


TEST(histogram_u8, single) {
	constexpr size_t s = 65;
	using T = uint8_t;
	T *data = (T *)malloc(s * sizeof(T));
	auto *cnt = (uint32_t *)malloc(256 * sizeof(uint32_t));
	memset(data, 0, s* sizeof(T));
	histogram_u8_1x(cnt, data, s);
	EXPECT_EQ(cnt[0], s);

	data[0] = 1;
	histogram_u8_1x(cnt, data, s);
	EXPECT_EQ(cnt[0], s-1);
	EXPECT_EQ(cnt[1], 1);

	free(data); free(cnt);
}

TEST(histogram_u8_4x, single) {
	constexpr size_t s = 65;
	using T = uint8_t;
	T *data = (T *)malloc(s * sizeof(T));
	auto *cnt = (uint32_t *)malloc(256 * sizeof(uint32_t));
	memset(data, 0, s* sizeof(T));
	histogram_u8_4x(cnt, data, s);
	EXPECT_EQ(cnt[0], s);

	data[0] = 1;
	histogram_u8_4x(cnt, data, s);
	EXPECT_EQ(cnt[0], s-1);
	EXPECT_EQ(cnt[1], 1);

	free(data); free(cnt);
}

#ifdef USE_AVX2
TEST(histogram_u32_avx2, single) {
	constexpr size_t s = 64;
	constexpr size_t s2 = 2048 + 8;
	using T = uint32_t;
	T *data = (T *)malloc(s * sizeof(T));
	auto *cnt = (uint32_t *)malloc(s2 * sizeof(uint32_t));
	memset(data, 0, s*sizeof(T));
	memset(cnt, 0, s2*sizeof(uint32_t));
	for (size_t i = 0; i < s; ++i) { data[i] = 1; }
	avx2_histogram_u32(cnt, data, s);
	EXPECT_EQ(cnt[1], s);
	for (uint32_t i = 2; i < 256; ++i) {
		EXPECT_EQ(cnt[i], 0);
	}

	// reset
	cnt[0] = 0;
	rng((uint8_t *)data, s * sizeof(T));
	for (size_t i = 0; i < s; ++i) { data[i] = data[i] & 0xFF; }
	avx2_histogram_u32(cnt, data, s);

	free(data); free(cnt);
}
#endif
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
