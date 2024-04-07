#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

// TODO currently dev
#if 0
#include "container/rb_tree.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using K = uint64_t;
using V = uint64_t;
TEST(RBTree, first) {
	auto t = RB_Tree<K, V, K>{};

}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
#endif