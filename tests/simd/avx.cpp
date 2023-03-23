#include <gtest/gtest.h>
#include <cstdint>

#include "helper.h"
#include "random.h"
#include "simd/avx2.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#if __x86_64__
TEST(U256, uint32_t) {
	U256i kek;
	print_m256i_u32(kek);
}
#endif

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
