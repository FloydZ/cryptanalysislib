#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "../test.h"
#include "helper.h"
#include "random.h"
#include "container/vector_queue.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(ConstVectorQueue, simple) {
	using T = uint32_t;
	using Q = ConstVectorQueue<T>;
	constexpr static uint32_t N=10;
	Q q;

	for (uint32_t t = 0; t < 1000; t++) {
		for (uint32_t i=0; i<N; i++) {
			EXPECT_TRUE(q.push(i));
		}

		for (uint32_t i=0; i<N; i++) {
			EXPECT_EQ(i, q.front());
			q.pop();
		}
	}
}

/// full queue, move only type, clear
TEST(ConstVectorQueue, full_move_clear) {
	using T = std::unique_ptr<uint32_t>;
	ConstVectorQueue<T, 8> q;
	for (uint32_t i = 0; i < 8; i++) {
		EXPECT_TRUE(q.push(std::make_unique<uint32_t>(i)));
	}
	EXPECT_FALSE(q.push(std::make_unique<uint32_t>(8)));
	EXPECT_EQ(q.size(), 8u);

	for (uint32_t i = 0; i < 4; i++) {
		T t = std::move(q.front());
		q.pop();
		EXPECT_EQ(*t, i);
	}
	EXPECT_TRUE(q.push(std::make_unique<uint32_t>(8)));
	EXPECT_EQ(*q.back(), 8u);

	q.clear();
	EXPECT_TRUE(q.empty());
	EXPECT_TRUE(q.push(std::make_unique<uint32_t>(9)));
	EXPECT_EQ(*q.front(), 9u);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
