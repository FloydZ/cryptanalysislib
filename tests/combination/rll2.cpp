#include <cstddef>
#include <gtest/gtest.h>

#include "combination/rll2.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(bit_rll2, p1) {
	using T = uint64_t;
	bit_rll2<T> b{};
    T w;
    uint32_t ctr = 0;
	do {
		w = b.next();
		print_binary(w, 14);
        ctr += 1;
	} while (ctr < 100);
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
