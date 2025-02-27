#include <cstddef>
#include <gtest/gtest.h>

#include "combination/paren.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(enum_paren, simple) {
	using T = uint64_t;
	constexpr static uint32_t n = 10;
    enumeration_parenthesis<T> b;
	for (size_t i = 0; i < (1u << n); i++){
		const T W = b.next();
		print_binary(W, n);
	}
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
