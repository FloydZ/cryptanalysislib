#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "atomic/atomic.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(pipe, single_threaded_read_back) {
	constexpr size_t s = 32;
	using T = uint64_t;
	using P = SimplePipe<T>;
	P p{};

	 T data1[s] = {0};
	 T data2[s] = {0};
	for (uint32_t i = 0; i < s; i++) {data2[i] = i;}

	for (uint32_t i = 0; i < s; i++) {
		const bool b = p.write_front(data1[i]);
		EXPECT_EQ(b, true);
	}

	for (uint32_t i = 0; i < s; i++) {
		const bool b = p.read_back(data2[i]);
		EXPECT_EQ(b, true);
	}

	T error;
	const bool b = p.read_back(error);
	EXPECT_EQ(b, false);

	for (uint32_t i = 0; i < 32; i++) {
		EXPECT_EQ(data2[i], data1[i]);
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
