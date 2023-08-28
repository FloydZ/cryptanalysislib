#include <gtest/gtest.h>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#if __APPLE__
TEST(u64_2, a) {
	u64_2 a = u64_2{1,1};
	a = xorb(a, a);
	print(a);
}
#endif

TEST(uint8x32_t, alles) {
	uint8x32_t t1, t2, t3;
	t1.random();
	t1.print();
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
