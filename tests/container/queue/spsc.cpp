#include <gtest/gtest.h>
#include <iostream>

#include "container/queue/spsc_fixed_queue.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

// TODO actual testing
TEST(SPSC, first) {
	using T = uint32_t;
	constexpr size_t N = 1u << 10;
	auto Q = spsc_fixed_queue<T>(10);
	for (uint32_t i = 0; i < N; i++) {
		Q.push(i);
	}
	for (uint32_t i = 0; i < N; i++) {
		const auto p = Q.pop();
		std::cout << p << std::endl;
	}
}

TEST(SPSC, iterator) {
	using T = uint32_t;
	constexpr size_t N = 8;
	auto Q = spsc_fixed_queue<T>(N);
	for (uint32_t i = 0; i < N; i++) {
		Q.push(i);
	}
	for (const auto &p : Q) {
		std::cout << p << std::endl;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
