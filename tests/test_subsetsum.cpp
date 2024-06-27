#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "kAry_type.h"
#include "tree.h"
#include "matrix/fq_matrix.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// max n = 15
constexpr uint32_t n    = 15;
constexpr uint32_t q    = 1u << n;

using T 			= uint8_t;
using Matrix 		= FqMatrix<T, 1, n, q>;
using Value     	= kAryContainer_T<T, n, 2>;
using Label    		= kAry_Type_T<q>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

// unused ignore
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};

TEST(SubSetSum, Simple) {
	// even it says matrix. It is a simple row vector
	Matrix A;
	A.fill(0);

	// // log size
	constexpr size_t list_size = 10;
	static std::vector<uint64_t> tbl{{0, n}};
	Tree t{1, A, list_size, tbl, __level_filter_array};

	t[0].generate_base_random(1u << list_size, A);
	// t[1].generate_base_random(1u << list_size, A);
	t.join_stream(0);

	//EXPECT_EQ(1u << 20u, t[2].load());
}



int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
