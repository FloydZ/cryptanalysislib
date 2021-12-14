#include <gtest/gtest.h>
#include <cstdint>

#include <bitset>
// IMPORTANT: Define 'SSLWE_CONFIG_SET' before one include 'helper.h'.
#ifndef SSLWE_CONFIG_SET
#define SSLWE_CONFIG_SET
#define G_l                     0u                  // unused Parameter
#define G_k                     0u                  // unused Parameter
#define G_d                     0u                  // unused Parameter
#define G_n                     1100u               // MUST be > 256 s.t. the AVX optimisation works.
#define LOG_Q                   1u                  // unused Parameter
#define G_q                     1u                  // unused Parameter
#define G_w                     1u                  // unused Parameter

#define SORT_INCREASING_ORDER
#define VALUE_BINARY

static  std::vector<uint64_t>                     __level_translation_array{{0, G_n/4, G_n/2, G_n}};
constexpr std::array<std::array<uint8_t, 3>, 3>   __level_filter_array{{ {{0,0,0}}, {{0,0,0}}, {{0,0,0}} }};
#endif

#define USE_AVX2

// Hack for testing private functions (C++ god)
#define private public

#include "helper.h"
#include "label.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(AddAVX2, Full_Length_Zero) {
	BinaryContainer<> b1;
	BinaryContainer<> b2;
	BinaryContainer<> b3;

	b1.zero(); b2.zero(); b3.zero();

	BinaryContainer<>::add(b3, b1, b2, 0, G_n);
	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	b3.random();
	BinaryContainer<>::add(b3, b1, b2, 0, G_n);
	for (int j = 0; j < b3.size(); ++j) {
		//std::cout << j << "\n";
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(AddAVX2, Full_Length_One) {
	BinaryContainer<> b1;
	BinaryContainer<> b2;
	BinaryContainer<> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;
	BinaryContainer<>::add(b3, b1, b2, 0, G_n);
	EXPECT_EQ(1, b3[0]);

	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<>::add(b3, b1, b2, 0, G_n);
	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);

	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<>::add(b3, b1, b2, 0, G_n);
	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(AddAVX2, OffByOne_Lower_One) {
	BinaryContainer<> b1;
	BinaryContainer<> b2;
	BinaryContainer<> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;   // this should be ignored.
	BinaryContainer<>::add(b3, b1, b2, 1, G_n);
	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<>::add(b3, b1, b2, 1, G_n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);
	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<>::add(b3, b1, b2, 1, G_n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);

	for (int j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(AddAVX2, OffByOne_Higher_One) {
	BinaryContainer<> b1;
	BinaryContainer<> b2;
	BinaryContainer<> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[G_n-1] = true;   // this should be ignored.
	BinaryContainer<>::add(b3, b1, b2, 0, G_n - 1);
	for (int j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<>::add(b3, b1, b2, 0, G_n - 1);
	EXPECT_EQ(0, b3[G_n-1]);
	EXPECT_EQ(false, b3[G_n-1]);
	for (int j = 0; j < b3.size() - 1; ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<>::add(b3, b1, b2, 0, G_n - 1);
	EXPECT_EQ(0, b3[G_n-1]);
	EXPECT_EQ(false, b3[G_n-1]);

	for (int j = 1; j < b3.size() - 1; ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}
/* currently not implemented
TEST(Add_Full_LengthAVX2, Probabilistic) {
	// Tests the full length add functions which returns also the weight of the element
	BinaryContainer<> b1;
	BinaryContainer<> b2;
	BinaryContainer<> res;

	uint64_t weight = 0;
	b1.zero(); b2.zero(); res.zero();
	for (int i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		weight = BinaryContainer<>::add(res, b1, b2);
		EXPECT_EQ(weight, i+1);
		EXPECT_EQ(true, b1.is_equal(res, 0, b1.size()));
		EXPECT_EQ(false, b2.is_equal(res, 0, b1.size()));

		if (i+1 < b1.size())
			EXPECT_EQ(true, b2.is_equal(res, i+1, b1.size()));

		res.zero();
	}
}
*/

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif