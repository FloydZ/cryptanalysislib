#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <execution>

#include "algorithm/for_each.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;

TEST(for_each, single) {
	using T = uint64_t;
	constexpr size_t size = 10;
	std::vector<T> test;
    test.resize(size);
	for (size_t i = 0; i < size; ++i) { test[i] = i; }

	// 	- number of threads
	//  - dynamic threading on/off
	//  see:https://github.com/alugowski/poolSTL/blob/26d95b90aea7c36732a2df50df1c6fa26c96f93e/tests/poolstl_test.cpp#L223
	cryptanalysislib::for_each(par_if(true), test.begin(), test.end(), [](auto &in){
		const auto t = in * in;
		in = t;
	});

	for (size_t i = 0; i < size; ++i) {
		std::cout << test[i] << std::endl;
		ASSERT_EQ(test[i], i*i);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
