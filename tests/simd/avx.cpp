#include <gtest/gtest.h>
#include <cstdint>

#include "helper.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#if USE_AVX2
#include "simd/avx2.h"
TEST(Bla, uint32_t) {
}
#endif

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
