#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "dispatch.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


uint32_t test1(const uint32_t a) {
    uint32_t ret = 0;
    for (uint32_t i = 0; i < a; i++) {
    	// just make it expensive
        ret += sqrtf(i*a);
    }
    return ret;
}

// idk: something faster
uint32_t test2(const uint32_t a) {
    uint32_t ret = 0;
	ret += a*a;
    return ret;
}

constexpr static size_t s = 1<<10;

TEST(dispatch_benchmark, simple) {
    const size_t t1 = genereric_bench(test1, s);
    const size_t t2 = genereric_bench(test2, s);
    EXPECT_GT(t1, t2);
}

TEST(dispatch, simple) {
    using F = uint32_t(*)(uint32_t);
	F out;
    std::vector<F> fs{test1, test2};

	const uint32_t t = s;
    const size_t pos = genereric_dispatch(out, fs, t);
    EXPECT_GT(pos, 0);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
