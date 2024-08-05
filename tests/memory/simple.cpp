#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "memory/memory.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#define T uint8_t
#define size 1000
#define A Memory_uint8_1000
#include "test.h"
#undef T
#undef size
#undef A

#define T uint16_t
#define size 1000
#define A Memory_uint16_1000
#include "test.h"
#undef T
#undef size
#undef A

#define T uint32_t
#define size 1000
#define A Memory_uint32_1000
#include "test.h"
#undef T
#undef size
#undef A

#define T uint64_t
#define size 1000
#define A Memory_uint64_1000
#include "test.h"
#undef T
#undef size
#undef A

#define T uint8_t
#define size 100
#define A Memory_uint8_100
#include "test.h"
#undef T
#undef size
#undef A

#define T uint16_t
#define size 100
#define A Memory_uint16_100
#include "test.h"
#undef T
#undef size
#undef A

#define T uint32_t
#define size 100
#define A Memory_uint32_100
#include "test.h"
#undef T
#undef size
#undef A

#define T uint64_t
#define size 100
#define A Memory_uint64_100
#include "test.h"
#undef T
#undef size
#undef A

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
