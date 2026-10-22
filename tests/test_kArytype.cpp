#include <sstream>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "helper.h"
#include "random.h"
#include "container/kAry_type.h"
#include "math/math.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

//#define TESTSIZE (1u << 16u)
#define TESTSIZE 1000


#define PRIME 2
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_2
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

#define PRIME 3
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_3
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

#define PRIME 5
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_5
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

#define PRIME ((1u << 16))
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_116
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

#define PRIME ((1u << 16) + 1)
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_1161
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

#define PRIME ((1ull << 25) - 1ull)
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_1251
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

#define PRIME ((1ull << 34) - 1ull)
#define S kAry_Type_T<PRIME>
#define T kAry_Type_T_1341
#include "test_kArytype.h"
#undef PRIME
#undef T
#undef S

/// operator<< writes into the given stream (not std::cout)
TEST(kAryType, ostream) {
	using K = kAry_Type_T<13>;
	K a = 5;
	std::stringstream ss;
	ss << a;
	EXPECT_EQ(ss.str(), "1010 (5, 0x5)"); // lowest bit first
}

/// same arithmetic with a wider limb type
TEST(kAryType, big) {
	using K = kAry_Type_T<13>;
	using B = kAry_Type_T_big<uint64_t, 13>;
	static_assert(std::is_same_v<K::LimbType, uint8_t>);
	static_assert(std::is_same_v<B::LimbType, uint64_t>);
	for (uint64_t i = 0; i < 13; ++i) {
		for (uint64_t j = 0; j < 13; ++j) {
			const K a = i, b = j;
			const B c = i, d = j;
			EXPECT_EQ((a + b).value(), (c + d).value());
			EXPECT_EQ((a - b).value(), (c - d).value());
			EXPECT_EQ((a * b).value(), (c * d).value());
		}
	}

	std::stringstream ss;
	ss << B(5);
	EXPECT_EQ(ss.str(), "1010 (5, 0x5)");
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(nullptr));
	return RUN_ALL_TESTS();
}
