#include <gtest/gtest.h>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#if __APPLE__
TEST(u64_2, a) {
	u64_2 a = u64_2{1,1};
	a = xorb(a, a);
	print(a);
}
#endif

TEST(uint8x32_t, set) {
	uint8x32_t t1 = uint8x32_t::set1(0);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[i] , 0);
	}

	uint8x32_t t2 = uint8x32_t::set1(1);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t2.v8[i] , 1);
	}
}

TEST(uint8x32_t, logic) {
	const uint8x32_t t1 = uint8x32_t::set1(0);
	const uint8x32_t t2 = uint8x32_t::set1(1);
	uint8x32_t t3 = uint8x32_t::set1(2);

	t3 = t1 + t2;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 1);
	}

	t3 = t2 - t1;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 1);
	}

	t3 = t2 - t2;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 0);
	}

	t3 = t1 ^ t2;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 1);
	}

	t3 = t1 | t2;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 1);
	}

	t3 = t1 & t2;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 0);
	}

	t3 = ~t1;
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , uint8_t(-1u));
	}
}

TEST(uint8x32_t, random) {
	uint8x32_t t1;
	t1.random();

	bool atleast_one_not_zero = false;
	for (uint32_t i = 0; i < 32; ++i) {
		if (t1.v8[i] > 0) {
			atleast_one_not_zero = true;
			break;
		}
	}

	EXPECT_EQ(atleast_one_not_zero, true);
}

TEST(uint8x32_t, unalinged_load) {
	uint8x32_t t1;
	uint8_t data[32] = {0};
	t1.random();

	t1 = uint8x32_t::unaligned_load(data);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[0], 0u);
	}
}

TEST(uint8x32_t, alinged_load) {
	uint8x32_t t1;
	alignas(256) uint8_t data[32] = {0};
	t1.random();

	t1 = uint8x32_t::aligned_load(data);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[0], 0u);
	}
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
