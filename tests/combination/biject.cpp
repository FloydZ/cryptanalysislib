#include <cstddef>
#include <gtest/gtest.h>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"
#include "math/math.h"

using namespace cryptanalysislib;
using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(reverse, p1) {
	constexpr size_t n = 12;
	constexpr size_t p = 2;
	for (uint64_t i = 0; i < bc(n, p); ++i) {
		if (popcount::popcount(i) > p) { continue; }
		const size_t t = reverse_biject<uint64_t, 10, 2>(i);
		std::cout << i << " " << t << " out" << std::endl;
	}
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
