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


#define T uint8_t
#define size 127
#define A Memory_uint8_127
#include "test.h"
#undef T
#undef size
#undef A

#define T uint16_t
#define size 127
#define A Memory_uint16_127
#include "test.h"
#undef T
#undef size
#undef A

#define T uint32_t
#define size 127
#define A Memory_uint32_127
#include "test.h"
#undef T
#undef size
#undef A

#define T uint64_t
#define size 127
#define A Memory_uint64_127
#include "test.h"
#undef T
#undef size
#undef A


#define T uint8_t
#define size 129
#define A Memory_uint8_129
#include "test.h"
#undef T
#undef size
#undef A

#define T uint16_t
#define size 129
#define A Memory_uint16_129
#include "test.h"
#undef T
#undef size
#undef A

#define T uint32_t
#define size 129
#define A Memory_uint32_129
#include "test.h"
#undef T
#undef size
#undef A

#define T uint64_t
#define size 129
#define A Memory_uint64_129
#include "test.h"
#undef T
#undef size
#undef A


#define T uint8_t
#define size (1u<<16)
#define A Memory_uint8_1u16
#include "test.h"
#undef T
#undef size
#undef A

#define T uint16_t
#define size (1u<<16)
#define A Memory_uint16_1u16
#include "test.h"
#undef T
#undef size
#undef A

#define T uint32_t
#define size (1u<<16)
#define A Memory_uint32_1u16
#include "test.h"
#undef T
#undef size
#undef A

#define T uint64_t
#define size (1u<<16)
#define A Memory_uint64_1u16
#include "test.h"
#undef T
#undef size
#undef A

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
