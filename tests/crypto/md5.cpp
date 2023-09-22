
#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>

#include "helper.h"
#include "random.h"
// TODO c#include "crypto/md5.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

//using namespace cryptanalysislib;

TEST(MD5, simple) {
// 	constexpr MD5 hash_value = MD5("Hello World");
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
