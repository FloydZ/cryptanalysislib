#include <cstddef>
#include <gtest/gtest.h>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"
#include "math/math.h"

using namespace cryptanalysislib;
using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t n = 100;

TEST(biject_simd, p1) {
	uint32x8_t a = uint32x8_t::setr(0,1,2,3,4,5,6,7);
	uint32x8_t rows1[1], rows2[1];

	biject_simd<n, 1>(a, rows1);
	biject_simd_lookup<n, 1>(a, rows2);

	for (int i = 0; i < 8; ++i) {
		EXPECT_EQ(rows1[0].v32[i], rows2[0].v32[i]);
	}
}

TEST(biject_simd, p2) {
	constexpr uint32_t p = 2;
	uint32x8_t a = uint32x8_t::setr(0,1,2,3,4,5,6,7);
	uint32x8_t rows1[p], rows2[p];

	biject_simd<n, p>(a, rows1);
	biject_simd_lookup<n, p>(a, rows2);

	for (uint32_t j = 0; j < 2; ++j) {
		for (uint32_t i = 0; i < 8; ++i) {
			EXPECT_EQ(rows1[j].v32[i], rows2[j].v32[i]);
		}
	}
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
