#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "random.h"
#include "triple.h"
#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// NOT implemented
// using Triple = triple<uint64_t, std::array<uint32_t, 2>, Label>;
// 
// TEST(Triple, first) {
// 	Triple a{};
// 	// std::cout << std::format("{} " , a) << "\n";
// }

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
