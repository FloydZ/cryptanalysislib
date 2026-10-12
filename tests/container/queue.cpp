#include <deque>
#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "../test.h"
#include "alloc/alloc.h"
#include "container/queue.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


/// fxt ring buffer queue: FIFO order across wrap arounds and growth
TEST(queue, ring_buffer_grow) {
	cryptanalysislib::queue<uint64_t, cryptanalysislib::allocator> q(3, 2);
	std::deque<uint64_t> e;
	uint64_t next = 0;
	for (uint32_t r = 0; r < 2000; ++r) {
		if (rand() % 3) {
			ASSERT_NE(q.push(next), 0u);
			e.push_back(next++);
		} else {
			uint64_t z = 0;
			const uint64_t n = q.pop(z);
			ASSERT_EQ(n, e.size());
			if (n) { ASSERT_EQ(z, e.front()); e.pop_front(); }
		}
		ASSERT_EQ(q.num(), e.size());
	}

	cryptanalysislib::queue<int> f(2); // never grows
	EXPECT_EQ(f.push(1), 1u);
	EXPECT_EQ(f.push(2), 2u);
	EXPECT_EQ(f.push(3), 0u);
	int z = 0;
	EXPECT_EQ(f.peek(z), 2u);
	EXPECT_EQ(z, 1);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
