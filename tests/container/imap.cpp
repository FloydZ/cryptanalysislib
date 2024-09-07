#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "container/imap.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO: more tests: https://github.com/billziss-gh/imap/blob/master/test/test.c
TEST(imap, first) {
	imap_tree_t tree;

	auto *slot = tree.lookup(0xA0000056);
	EXPECT_EQ(nullptr, slot);
	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);

	EXPECT_FALSE(imap_tree_t::hasval(slot));
	EXPECT_EQ(0, tree.getval(slot));
	tree.setval(slot, 0x56);
	EXPECT_TRUE(tree.hasval(slot));
	EXPECT_EQ(0x56, tree.getval(slot));

	slot = tree.lookup(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.delval(slot);
	EXPECT_FALSE(tree.hasval(slot));
	slot = tree.lookup(0xA0000056);
	EXPECT_EQ(nullptr, slot);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
