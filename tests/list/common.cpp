#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "matrix/fq_matrix.h"
#include "list/common.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t k = 100;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;

using MatrixT = uint8_t;
using Matrix 		= FqMatrix<MatrixT, n, k, q>;
using Value 		= kAryContainer_T<MatrixT, k, q>;
using Label 		= kAryContainer_T<MatrixT, n, q>;
using Element       = Element_T<Value, Label, Matrix>;
using List 			= MetaListT<Element>;


TEST(List, copy) {
	constexpr size_t size = 100;
	List L{size,1};
	List L1{1,1};
	L.zero();
	L1 = L;

	EXPECT_EQ(L1.size(), L.size());
	EXPECT_EQ(L1.load(), L.load());
	EXPECT_EQ(L1.threads(), L.threads());
	EXPECT_EQ(L1.thread_block_size(), L.thread_block_size());
	for (size_t i = 0; i < L1.size(); ++i) {
		EXPECT_EQ(L1.at(i).is_zero(), true);
	}
}

TEST(List, random_copy) {
	constexpr size_t size = 100;
	List L{size,1};
	List L1{50,1};
	L.zero();
	L1.random();
	for (size_t i = 0; i < L1.size(); ++i) {
		EXPECT_EQ(L1.at(i).is_zero(), false);
	}
	L1 = L;

	EXPECT_EQ(L1.size(), L.size());
	EXPECT_EQ(L1.load(), L.load());
	EXPECT_EQ(L1.threads(), L.threads());
	EXPECT_EQ(L1.thread_block_size(), L.thread_block_size());
	for (size_t i = 0; i < L1.size(); ++i) {
		EXPECT_EQ(L1.at(i).is_zero(), true);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
