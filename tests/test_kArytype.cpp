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

// dev
// TEST(T, dev_big) {
// 	kAry_Type_T_big<>
// }
int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(nullptr));
	return RUN_ALL_TESTS();
}
