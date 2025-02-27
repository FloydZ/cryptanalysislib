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
	constexpr static T m = -1ull;
	bit_fibgray<T, n> b;
	T W;
	do {
		W = b.next();
		print_binary(W, n);
	} while (m != W);
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
