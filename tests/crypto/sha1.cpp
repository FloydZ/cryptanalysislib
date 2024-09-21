#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "random.h"

// TODO not implemented
// #include "crypto/sha1.h"
// 
// 
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
//	constexpr uint8_t data[3] = {"He"};
// 	constexpr SHA1 hash_value = SHA1(data);
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
