#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "../test.h"
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

typedef struct _handle_t {
	uint64_t lhead;
} handle_t;

TEST(lock_free_queue, test) {
	handle_t th = handle_t();

	auto q = lock_free_queue<T>();
	T val = 0;
	q.lfring_ptr_init_lhead(&th.lhead);
	q.enqueue(&val, false, false, &th.lhead);

	T val2 = 1;
	q.enqueue(&val2, false, false, &th.lhead);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
