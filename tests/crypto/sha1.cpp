#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "random.h"

#include "crypto/sha1.h"
using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
// 
// using namespace cryptanalysislib;
// 
TEST(SHA1, simple) {
    static_assert("abc"_sha1       == "a9993e364706816aba3e25717850c26c9cd0d89d"_hex_bytes);
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
