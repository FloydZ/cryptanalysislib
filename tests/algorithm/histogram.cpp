#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/histogram.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


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
	using T = uint16_t;
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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
