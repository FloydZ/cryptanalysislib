#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/search.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


template <typename T>
class Search : public testing::Test {};

TYPED_TEST_SUITE_P(Search);

TYPED_TEST_P(Search, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);
    const auto r1 = cryptanalysislib::search(in.begin(), in.end(), 1);
	const auto t1 = std::distance(in.begin(), r1);
	EXPECT_EQ(t1, 0);

	const auto r2 = cryptanalysislib::search(in.begin(), in.end(), 0);
	const auto t2 = std::distance(in.begin(), r2);
	EXPECT_EQ((size_t)t2, s);
}

TYPED_TEST_P(Search, simple_n) {
    constexpr static size_t n = 10;
    constexpr static size_t s = 1000;
    std::vector<TypeParam> in; in.resize(s);
	std::fill(in.begin(), in.end() - n, 2);
	std::fill(in.begin()+s-n, in.end(), 1);
    
    const auto r1 = cryptanalysislib::search_n(in.begin(), in.end(), 1, n);
	const auto t1 = std::distance(in.begin(), r1);
	EXPECT_EQ(t1, s - n);

	const auto r2 = cryptanalysislib::search(in.begin(), in.end(), 0 , n);
	const auto t2 = std::distance(in.begin(), r2);
	EXPECT_EQ((size_t)t2, s);
}

//TYPED_TEST_P(Search, multithreading) {
//    constexpr static size_t s = 10000;
//    std::vector<TypeParam> in; in.resize(s);
//    std::fill(in.begin(), in.end(), 1);
//
//    const auto r1 = cryptanalysislib::search(par_if(true), in.begin(), in.end(), 1);
//	const auto t1 = std::distance(in.begin(), r1);
//	EXPECT_EQ(t1, 0);
//}

REGISTER_TYPED_TEST_SUITE_P(Search, simple);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Search, MyTypes);


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
