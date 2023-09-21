#include <gtest/gtest.h>

#include "random.h"
#include "helper.h"
#include "alloc/alloc.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(StackAllocator, Simple) {
	StackAllocator<16> s;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




