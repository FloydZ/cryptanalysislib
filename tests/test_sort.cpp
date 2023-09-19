#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "matrix/fq_matrix.h"
#include "sort.h"

constexpr uint64_t ListSize = 15;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace std;


TEST(Std_Sort_Binary_Container, SingleLimb_Simple) {
	constexpr uint64_t size = 63;
	const uint64_t k_lower = 0;
	const uint64_t k_higher = size;

	using BinaryContainer = BinaryContainer<size>;

	vector<BinaryContainer> data;
	data.resize(ListSize);
	for (int i = 0; i < ListSize; ++i) {
		data[i].random();
	}

	Std_Sort_Binary_Container<BinaryContainer>::sort(data, k_lower, k_higher);

	for(uint64_t i = 0; i < ListSize-1; i++){
#if defined(SORT_INCREASING_ORDER)
		EXPECT_EQ(true, data[i].is_lower(data[i+1], k_lower, k_higher));
#else
		EXPECT_EQ(true, data[i].is_greater(data[i+1], k_lower, k_higher));
#endif
	}
}

TEST(Std_Sort_Binary_Container, MultiLimb_Simple) {
	constexpr uint64_t size = 255;
	const uint64_t k_lower = 0;
	const uint64_t k_higher = size;
	using BinaryContainer = BinaryContainer<size>;


	vector<BinaryContainer> data;
	data.resize(ListSize);
	for (int i = 0; i < ListSize; ++i) {
		data[i].random();
	}

	Std_Sort_Binary_Container<BinaryContainer>::sort(data, k_lower, k_higher);

	for(uint64_t i = 0; i < ListSize-1; i++){
#if defined(SORT_INCREASING_ORDER)
		EXPECT_EQ(true, data[i].is_lower(data[i+1], k_lower, k_higher));
#else
		EXPECT_EQ(true, data[i].is_greater(data[i+1], k_lower, k_higher));
#endif
	}

}

uint64_t Hash(uint64_t a){
	return a;
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
