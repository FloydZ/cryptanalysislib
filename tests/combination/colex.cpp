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
	for (size_t i = 0; i < bc(n, w); i++){
		const T W = b.next();
		print_binary(W, n);
	}
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
