#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

#include "algorithm/subsetsum.h"

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
//using Value     	= kAryPackedContainer_T<T, n, 2>;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;


TEST(SubSetSum, dissection) {
	Label::info();
	Matrix::info();

	Matrix AT; AT.random();

	List out{1<<n};
	// NOTE: the base lists of the 4-way dissection enumerate weight `n/8`
	// 	on each quarter of the coordinates, so the solution is planted with
	// 	exactly `n/8` ones in each quarter.
	Label target; target.zero();
	for (uint32_t qd = 0; qd < 4; ++qd) {
		std::vector<uint32_t> idx(n/4);
		for (uint32_t i = 0; i < n/4; ++i) { idx[i] = qd*(n/4) + i; }
		for (uint32_t i = 0; i < n/8; ++i) {
			const uint32_t j = i + (uint32_t)(rng() % (n/4 - i));
			cryptanalysislib::swap(idx[i], idx[j]);
			Label::add(target, target, AT[0][idx[i]]);
		}
	}

	Tree::constexpr_dissection4<0, n/4, n>(out, target, AT);

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
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
