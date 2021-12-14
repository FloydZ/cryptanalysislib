#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <bitset>


#include "helper.h"
#include "label.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(Access, get_data_non_const_access) {
	Label_T<BinaryContainer<n>> l;

	l.zero();

	for (int i = 0; i < n; ++i) {
		EXPECT_EQ(0, l.data()[i]);

		l.data()[i] = true;
		EXPECT_EQ(1, l.data()[i]);

	}
}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
#endif