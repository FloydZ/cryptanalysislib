#include <cstddef>
#include <gtest/gtest.h>

#include "combination/colex.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(bit_subset, p1) {
	using T = uint64_t;
	constexpr static uint32_t n = 10;
	constexpr static uint32_t w = 3;
	enumeration_colex<T, n, w> b;
	size_t ctr = 0;
	do {
		print_binary(b.current(), n);
		ctr += 1;
	} while (b.next());
	EXPECT_EQ(ctr, bc(n, w));
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
