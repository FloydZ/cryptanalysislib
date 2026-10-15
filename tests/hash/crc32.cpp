#include <vector>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>

#include "hash/hash.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;
using namespace cryptanalysislib::hash;

TEST(Hash, simple) {

}

#if defined(USE_PCLMUL) && defined(USE_SSE41)
TEST(crc32, sse42) {
	constexpr static size_t size = 1024;
	auto *data = (uint8_t *)malloc(size);
	rng(data, size);

	for (uint32_t i = 0; i < size; ++i) {
		const uint32_t t1 = crc32(data, i, 0);
		const uint32_t t2 = sse42_crc32(data, i, 0);
		EXPECT_EQ(t1, t2);
		// std::cout << i << " " << t1 << " " << t2 << std::endl;
	}
}
#endif
/// the check value of CRC32 and all lengths: table, SSE4.2 and dispatch agree
TEST(crc32, check_value_and_dispatch) {
	const char *check = "123456789";
	EXPECT_EQ(crc32((const uint8_t *)check, 9, 0), 0xCBF43926u);
	EXPECT_EQ(crc32_dispatch((const uint8_t *)check, 9, 0), 0xCBF43926u);

	constexpr static size_t size = 1024;
	std::vector<uint8_t> data(size);
	for (auto &x : data) { x = uint8_t(rand()); }
	for (size_t i = 0; i <= size; ++i) {
		const uint32_t e = crc32(data.data(), i, 0x1234u);
		EXPECT_EQ(crc32_dispatch(data.data(), i, 0x1234u), e) << i;
#if defined(USE_PCLMUL) && defined(USE_SSE41)
		EXPECT_EQ(sse42_crc32(data.data(), i, 0x1234u), e) << i;
#endif
	}
	// incremental == one shot
	EXPECT_EQ(crc32(data.data() + 100, size - 100, crc32(data.data(), 100, 0)), crc32(data.data(), size, 0));
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
    return RUN_ALL_TESTS();
}

