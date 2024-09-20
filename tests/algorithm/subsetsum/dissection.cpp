#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n    = 16ul;
constexpr uint32_t q    = (1ul << n);

using T 			= uint64_t;
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= BinaryContainer<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;


TEST(SubSetSum, dissection) {
	Label::info();
	Matrix::info();

	Matrix AT; AT.random();
	std::cout << AT;

	List out{1<<n};
	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, AT[0][weights[i]]);
	}

	Tree::dissection4(out, target, AT);

	EXPECT_GE(out.load(), 1);
	for (size_t i = 0; i < out.load(); ++i) {
		target.print_binary();
		out[i].label.print_binary();
		// std::cout << target << ":" << out[i].label << std::endl;
		Label tmp;
		AT.mul(tmp, out[i].value);

		EXPECT_EQ(target.is_equal(tmp), true);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
