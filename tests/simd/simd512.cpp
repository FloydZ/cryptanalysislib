#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(uint8x64_t, TTrandom) {
	uint8x64_t t1 = uint8x64_t::random();

	uint32_t atleast_one_not_zero = false;
	for (uint32_t i = 0; i < 32; ++i) {
		if (t1.v8[i] > 0) {
			atleast_one_not_zero = true;
		//	break;
		}
	}

	ASSERT_EQ(atleast_one_not_zero, true);
}

TEST(uint8x64_t, set1) {
	uint8x64_t t1 = uint8x64_t::set1(0);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[i] , 0);
	}

	uint8x64_t t2 = uint8x64_t::set1(1);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t2.v8[i] , 1);
	}
}

TEST(uint8x64_t, set) {
	uint32_t pos = 21;
	uint8x64_t t1 = uint8x64_t::set(0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	for (uint32_t i = 0; i < 64; ++i) {
		if (i == pos){
			EXPECT_EQ(t1.v8[i] , 1);
			continue;
		}
		EXPECT_EQ(t1.v8[i] , 0);
	}

	uint8x64_t t2 = uint8x64_t::setr(0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	for (uint32_t i = 0; i < 64; ++i) {
		if (i == (31-pos)){
			EXPECT_EQ(t2.v8[i] , 1);
			continue;
		}
		EXPECT_EQ(t2.v8[i] , 0);
	}
}

TEST(uint8x64_t, unalinged_load) {
	uint8_t data[32] = {0};

	uint8x64_t t1 = uint8x64_t::unaligned_load(data);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[i], 0u);
	}
}

TEST(uint8x64_t, alinged_load) {
	alignas(256) uint8_t data[32] = {0};
	uint8x64_t t1 = uint8x64_t::aligned_load(data);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[i], 0u);
	}
}

TEST(uint8x64_t, unalinged_store) {
	uint8x64_t t1 = uint8x64_t::random();
	uint8_t data[32] = {0};

	uint8x64_t::unaligned_store(data, t1);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[i], data[i]);
	}
}

TEST(uint8x64_t, alinged_store) {
	uint8x64_t t1 = uint8x64_t::random();
	alignas(256) uint8_t data[32] = {0};

	uint8x64_t::aligned_store(data, t1);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t1.v8[i], data[i]);
	}
}
TEST(uint8x64_t, logic) {
	const uint8x64_t t1 = uint8x64_t::set1(0);
	const uint8x64_t t2 = uint8x64_t::set1(1);
	uint8x64_t t3 = uint8x64_t::set1(2);

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

	t3 = uint8x64_t::mullo(t1, t2);
	for (uint32_t i = 0; i < 32; ++i) {
		EXPECT_EQ(t3.v8[i] , 0);
	}

	// t3 = uint8x64_t::slli(t1, 1);
	// for (uint32_t i = 0; i < 32; ++i) {
	// 	EXPECT_EQ(t3.v8[i] , 0);
	// }

	// t3 = uint8x64_t::slli(t2, 1);
	// for (uint32_t i = 0; i < 32; ++i) {
	// 	EXPECT_EQ(t3.v8[i] , 2);
	// }
}

//TEST(uint8x64_t, slri) {
//	for (uint8_t j = 0; j < 8; j++) {
//		const uint8x64_t t1 = uint8x64_t::set1(1u << j);
//		const uint8x64_t t2 = uint8x64_t::srli(t1, j);
//		for (uint32_t i = 0; i < 32; ++i) {
//			EXPECT_EQ(t2.v8[i], 1);
//		}
//	}
//
//	/// special case for j = 8
//	const uint8x64_t t1 = uint8x64_t::set1((1u << 7u) - 1u);
//	const uint8x64_t t2 = uint8x64_t::srli(t1, 8);
//	for (uint32_t i = 0; i < 32; ++i) {
//		EXPECT_EQ(t2.v8[i], 0);
//	}
//}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
