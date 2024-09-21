#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "../test.h"
#include "helper.h"
#include "random.h"
// #include "container/queue.h"

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


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
