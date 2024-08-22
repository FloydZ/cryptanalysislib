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

TEST(Hash, bits) {
	using T = uint64_t;
	T d[2] = {-1ull, -1ull};
	T r = Hash<T, 0, 10, 2>::hash(d);
	T mask = (1u<<10u) -1u;
	EXPECT_EQ(r, mask);

	r = Hash<T, 60, 70, 2>::hash(d);
	mask = (1u<<10u) -1u;
	EXPECT_EQ(r, mask);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
    return RUN_ALL_TESTS();
}
