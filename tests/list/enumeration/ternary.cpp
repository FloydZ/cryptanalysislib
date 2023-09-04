#include <gtest/gtest.h>

#include "helper.h"
#include "random.h"
#include "list/enumeration/ternary.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(uint8x32_t, set) {
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
