#include <gtest/gtest.h>

#include "alloc/cache.h"
#include "helper.h"
#include "omp.h"


using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

#define THREADS 32

class TestStruct {
	uint64_t tmp = 0;
	uint64_t *ptr = nullptr;
};

TEST(AllocationMap, AllocateSimple) {
	CacheAllocator<TestStruct> allocator;
	TestStruct *ptr = allocator.allocate();
	EXPECT_NE(ptr, nullptr);
}

TEST(AllocationMap, Allocation) {
	CacheAllocator<TestStruct> allocator;
	for (size_t i = 0; i < 10000; ++i) {
		TestStruct *ptr = allocator.allocate();
		EXPECT_NE(ptr, nullptr);
	}
}

TEST(AllocationMap, DeallocateSimple) {
	CacheAllocator<TestStruct> allocator;
	EXPECT_EQ(allocator.size(), 0);
	TestStruct *ptr = allocator.allocate();
	EXPECT_EQ(allocator.size(), 1);
	TestStruct *ptr1 = allocator.allocate();
	EXPECT_EQ(allocator.size(), 2);
	EXPECT_NE(ptr, nullptr);
	EXPECT_NE(ptr1, nullptr);

	EXPECT_EQ(allocator.deallocate(nullptr), false);
	EXPECT_EQ(allocator.size(), 2);
	EXPECT_EQ(allocator.deallocate(ptr), true);
	EXPECT_EQ(allocator.size(), 1);
	EXPECT_EQ(allocator.deallocate(ptr1), true);
	EXPECT_EQ(allocator.size(), 0);
}

TEST(AllocationMap, SimpleMultithreaded) {
	CacheAllocator<TestStruct> allocator;
	TestStruct *t[THREADS];

	#pragma omp parallel default(none) shared(allocator, t) num_threads(THREADS)
	{
		TestStruct *ptr = allocator.allocate();
		EXPECT_NE(ptr, nullptr);
		const uint32_t pid = omp_get_thread_num();
		ASSERT(pid < THREADS);
		t[pid] = ptr;
	}

	for (uint32_t i = 0; i < THREADS; ++i) {
		for (uint32_t u = 0; u < THREADS; ++u) {
			if (i == u) {
				continue;
			}

			EXPECT_NE(t[i], t[u]);
			if (t[i] != t[u]) {
				break;
			}
		}
	}
}

TEST(AllocationMap, SingleThreadMultiple) {
	CacheAllocator<TestStruct> allocator;
	for (uint32_t i = 0; i < 65; ++i) {
		TestStruct *ptr = allocator.allocate();
		EXPECT_NE(ptr, nullptr);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
