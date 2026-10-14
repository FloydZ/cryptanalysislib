#include <cstddef>
#include <gtest/gtest.h>

#include "combination/fibonacci_gray.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(bit_subset, p1) {
	using T = uint64_t;
	constexpr static uint32_t n = 5;
	bit_fibgray<T, n> b;
	uint32_t ctr = 0;
	do {
		print_binary(b.current(), n);
		ctr += 1;
	} while (b.next());
	// F(n+2) Fibonacci words of length n
	EXPECT_EQ(ctr, 13u);
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
