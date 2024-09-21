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

TEST(Hash, simple) {

}

#ifdef USE_AVX2
TEST(crc32, sse42) {
	constexpr static size_t size = 1024;
	auto *data = (uint8_t *)malloc(size);
	rng(data, size);

	for (uint32_t i = 64; i < size; ++i) {
		const uint32_t t1 = crc32(data, i, 0);
		const uint32_t t2 = sse42_crc32(data, i, 0);
		EXPECT_EQ(t1, t2);
		// std::cout << i << " " << t1 << " " << t2 << std::endl;
	}
}
#endif
int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
    return RUN_ALL_TESTS();
}

