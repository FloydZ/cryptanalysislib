#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>

#include "helper.h"
#include "binary.h"

#define SORT_INCREASING_ORDER

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// Hack for testing private functions (C++ god)
//#define private public

#ifdef USE_AVX2

TEST(AddAVX2, Full_Length_Zero) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	b3.random();
	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		//std::cout << j << "\n";
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(AddAVX2, Full_Length_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;
	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	EXPECT_EQ(1, b3[0]);

	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);

	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(AddAVX2, OffByOne_Lower_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;   // this should be ignored.
	BinaryContainer<n>::add(b3, b1, b2, 1, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 1, n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);
	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 1, n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);

	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(AddAVX2, OffByOne_Higher_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[n-1] = true;   // this should be ignored.
	BinaryContainer<n>::add(b3, b1, b2, 0, n - 1);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n - 1);
	EXPECT_EQ(0, b3[n-1]);
	EXPECT_EQ(false, b3[n-1]);
	for (uint32_t j = 0; j < b3.size() - 1; ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n - 1);
	EXPECT_EQ(0, b3[n-1]);
	EXPECT_EQ(false, b3[n-1]);

	for (uint32_t j = 1; j < b3.size() - 1; ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

#endif

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
