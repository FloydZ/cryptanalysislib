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
			q.push(i);
		}

		for (uint32_t i=0; i<N; i++) {
			EXPECT_EQ(i, q.front());
			q.pop();
		}
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
