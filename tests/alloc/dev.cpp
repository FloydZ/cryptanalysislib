#include <gtest/gtest.h>

#include "random.h"
#include "helper.h"
#include "alloc/alloc.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(StackAllocator, Simple) {
	constexpr size_t size = 16;
	StackAllocator<size> s;
	Blk b = s.allocate(size);
	auto *ptr = (uint8_t *)b.ptr;
	for(size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}

	ASSERT_EQ(s.owns(b), true);
	Blk b2{((uint8_t *)b.ptr)+size, b.len};
	ASSERT_EQ(s.owns(b2), false);
	s.deallocateAll();

	ASSERT_EQ(s.owns(b),  false);
	ASSERT_EQ(s.owns(b2), false);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




