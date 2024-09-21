#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>

#include "hash/hash.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Hash, simple) {

}

#ifdef USE_AVX2
TEST(Adler32, avx) {
	constexpr static size_t size = 1024;
	auto *data = (uint8_t *)malloc(size);
	rng(data, size);

	for (uint32_t i = 64; i < size; ++i) {
		const uint32_t t1 = adler32(0, data, i);
		const uint32_t t2 = avx2_adler32(0, data, i);
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

