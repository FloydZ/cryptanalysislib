#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
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

//TEST(Bucket_Sort_Binary_Container_Single_Limb, SingleLimb_Simple) {
//	constexpr uint64_t size = 10;
//	const uint64_t k_lower = 0;
//	const uint64_t k_higher = size;
//	using BinaryContainer = BinaryContainer<size>;
//
//
//	vector<BinaryContainer> data;
//	data.resize(ListSize);	for (int i = 0; i < ListSize; ++i) {
//		data[i].random();
//	}
//
//	Bucket_Sort_Binary_Container_Single_Limb<BinaryContainer>::sort(data, k_lower, k_higher);
//
//	for(uint64_t i = 0; i < ListSize-1; i++){
//#if defined(SORT_INCREASING_ORDER)
//		EXPECT_EQ(true, data[i].is_lower(data[i+1], k_lower, k_higher));
//#else
//		EXPECT_EQ(true, data[i].is_greater(data[i+1], k_lower, k_higher));
//#endif
//	}
//}

//TEST(Shell_Sort_Binary_Container_Single_Limb, SingleLimb_Simple) {
//	constexpr uint64_t size = 11;
//	const uint64_t k_lower = 0;
//	const uint64_t k_higher = size;
//	using BinaryContainer = BinaryContainer<size>;
//
//	vector<BinaryContainer> data;
//	data.resize(ListSize);
//	for (int i = 0; i < ListSize; ++i) {
//		data[i].random();
//	}
//
//	Shell_Sort_Binary_Container_Single_Limb<BinaryContainer>::sort(data, k_lower, k_higher);
//
//	for(uint64_t i = 0; i < ListSize-1; i++){
//#if defined(SORT_INCREASING_ORDER)
//		EXPECT_EQ(true, data[i].is_lower(data[i+1], k_lower, k_higher));
//#else
//		EXPECT_EQ(true, data[i].is_greater(data[i+1], k_lower, k_higher));
//#endif
//	}
//}

/* NOT IMplemented
TEST(Radix_Sort_Binary_Container_Single_Limb, SingleLimb_Simple) {
	constexpr uint64_t size = 11;
	const uint64_t k_lower = 0;
	const uint64_t k_higher = size;
	using BinaryContainer = BinaryContainer<size>;

	vector<BinaryContainer> data;
	data.resize(ListSize);
	for (int i = 0; i < ListSize; ++i) {
		data[i].random();
	}

	Radix_Sort_Binary_Container_Single_Limb<BinaryContainer>::sort(data, k_lower, k_higher);

	for(uint64_t i = 0; i < ListSize-1; i++){
#if defined(SORT_INCREASING_ORDER)
		//EXPECT_EQ(true, data[i].is_lower(data[i+1], k_lower, k_higher));
#else
		EXPECT_EQ(true, data[i].is_greater(data[i+1], k_lower, k_higher));
#endif
	}
}
 */

uint64_t Hash(uint64_t a){
	return 0;
}
TEST(ParallelBucketSort, InterpolationSearch) {
	using DecodingValue = Value_T<BinaryContainer<50>>;
	using DecodingLabel = Label_T<BinaryContainer<100>>;
	using DecodingMatrix = mzd_t *;
	using DecodingElement = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
	using DecodingList = List_T<DecodingElement>;

	constexpr static uint64_t lmask = (1 << 12)-1;
	constexpr static ConfigParallelBucketSort config{0, 2, 12, 100001, 4, 1, 1, 0, 0, 0, 0};

	ParallelBucketSort<config, DecodingList, uint64_t, uint64_t, &Hash> hm;

	uint64_t npos[1];
	// insert a view random elements.
	for (uint64_t j = 0; j < 10001; ++j) {
		uint64_t data = random() & lmask;
		npos[0] = j;
		hm.insert(data, npos, 0);
	}

	hm.sort(0);
	// hm.print(0, 100);
	uint64_t load;
	for (uint64_t j = 0; j < 2000; ++j) {
		uint64_t data = hm.__buckets[j].first;
		uint64_t pos = hm.find(data, load);
		if (pos > j) {
			std::cout << "e\n";
		}
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
