#include <cstddef>
#include <gtest/gtest.h>

#include "combination/bit_subset.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(bit_subset, p1) {
	using T = uint64_t;
	T W, V = 0b11010000100001;
	bit_subset_T<T> b(V);
	do {
		W = b.next();
		print_binary(W, 14);
	} while (V != W);
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
