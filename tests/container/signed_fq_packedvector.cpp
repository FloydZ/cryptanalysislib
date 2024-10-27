#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_packed_vector.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(sign, simple) {
	using Fq = FqPackedVector<1, 3, uint32_t, false>;
	Fq t1(1);
	Fq t2(-1);
	std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
