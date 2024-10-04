#include "dispatch.h"
#include <gtest/gtest.h>
#include <iostream>

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
        ret += i*a;
    }
    return ret;
}

// idk: something faster
uint32_t test2(const uint32_t a) {
    uint32_t ret = 0;
    for (uint32_t i = 0; i < a; i+=16) {
        ret += i*a;
    }
    return ret;
}

constexpr static size_t s = 1<<10;

TEST(dispatch_benchmark, simple) {
    const size_t t1 = genereric_bench(test1, s);
    const size_t t2 = genereric_bench(test2, s);
    EXPECT_GT(t1, t2);
}

// TODO not implemented
TEST(dispatch, simple) {
    using F = uint32_t(uint32_t);
    //F* fs[] = {test1, test2};
    // std::vector<F> Fs{{test1, test2}};
    F *out = nullptr;
    uint32_t args[] = {s, s};
    // genereric_dispatch(out, fs, args, 2);

}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
