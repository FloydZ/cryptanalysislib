#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "../test.h"
#include "container/triple.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr static ConfigTriple configTriple{};
using Triple = triple<uint64_t, uint32_t, uint16_t, configTriple>;

TEST(Triple, first) {
	const Triple a{};
	EXPECT_TRUE((a.first == 0) && (a.second == 0) && (a.third == 0));

	const Triple b{1, 2, 3};
	EXPECT_TRUE((b.first == 1) && (b.second == 2) && (b.third == 3));
	EXPECT_TRUE(b == Triple(1, 2, 3));
	EXPECT_FALSE(a == b);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
