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

// APPLE SUCKS...
#ifndef __APPLE__
TEST(for_each, single) {
	using T = uint64_t;
	constexpr size_t size = 10;
	std::vector<T> test(size);
	for (size_t i = 0; i < size; ++i) { test[i] = i; }

	// TODO write cusom execution policies, which allow for
	// 	- number of threads
	//  - dynamic threading on/off
	//  see:https://github.com/alugowski/poolSTL/blob/26d95b90aea7c36732a2df50df1c6fa26c96f93e/tests/poolstl_test.cpp#L223
	cryptanalysislib::for_each(std::execution::seq, test.begin(), test.end(), [](auto &in){
		const auto t = in * in;
		in = t;
	});

	for (size_t i = 0; i < size; ++i) { std::cout << test[i] << std::endl; }
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
