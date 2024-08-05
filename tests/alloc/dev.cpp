#include <gtest/gtest.h>

#include "alloc/alloc.h"
#include "helper.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(StackAllocator, Simple) {
	constexpr size_t size = 16;
	StackAllocator<size> s;
	Blk b = s.allocate(size);
	ASSERT_EQ(b.valid(), true);

	auto *ptr = (uint8_t *) b.ptr;
	for (size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}

	ASSERT_EQ(s.owns(b), true);
	Blk b2{((uint8_t *) b.ptr) + size, b.len};
	ASSERT_EQ(s.owns(b2), false);
	s.deallocateAll();

	ASSERT_EQ(s.owns(b), false);
	ASSERT_EQ(s.owns(b2), false);
}

TEST(FreeListAllocator, Simple) {
	constexpr size_t total_size = 256;
	constexpr size_t size = 16;
	FreeListAllocator<StackAllocator<total_size>, size> s;
	Blk b = s.allocate(size);
	ASSERT_EQ(b.valid(), true);

	auto *ptr = (uint8_t *) b.ptr;
	for (size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}

	// checking some basics
	ASSERT_EQ(s.owns(b), true);
	s.deallocateAll();
	ASSERT_EQ(s.owns(b), true);

	// checking the free list, while debugging you should see, that in
	// the last loop the memory is not allocated anymore. But reused
	// from the FreeList
	Blk bb[total_size / size];
	for (uint32_t i = 0; i < (total_size / size); ++i) {
		bb[i] = s.allocate(size);
	}
	for (uint32_t i = 0; i < (total_size / size); ++i) {
		s.deallocate(bb[i]);
	}
	for (uint32_t i = 0; i < (total_size / size); ++i) {
		bb[i] = s.allocate(size);
	}
}

TEST(AffixAllocator, Simple) {
	constexpr size_t size = 16;
	struct TestStruct {
		uint64_t tmp;
	};
	AffixAllocator<StackAllocator<1024>, TestStruct> s;
	Blk b = s.allocate(size);
	ASSERT_EQ(b.valid(), true);

	auto *ptr = (uint8_t *) b.ptr;
	for (size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}

	ASSERT_EQ(s.owns(b), true);
	s.deallocate(b);
	ASSERT_EQ(s.owns(b), false);
	Blk b2{((uint8_t *) b.ptr) - size, b.len};
	ASSERT_EQ(s.owns(b2), false);
	s.deallocateAll();

	ASSERT_EQ(s.owns(b), false);
	ASSERT_EQ(s.owns(b2), false);
}


TEST(Segregator, Simple) {
	constexpr size_t size = 16;
	Segregator<FreeListAllocator<StackAllocator<1024>, 16>,
	           StackAllocator<4096>,
	           128>
	        s;
	Blk b = s.allocate(size);
	ASSERT_EQ(b.valid(), true);

	auto *ptr = (uint8_t *) b.ptr;
	for (size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}

	ASSERT_EQ(s.owns(b), true);
	s.deallocate(b);// FreeList Deallocate
	ASSERT_EQ(s.owns(b), true);
	Blk b2{((uint8_t *) b.ptr) - size, b.len};
	ASSERT_EQ(s.owns(b2), true);// well technically not true
	s.deallocateAll();

	// ASSERT_EQ(s.owns(b),  false);
	// ASSERT_EQ(s.owns(b2), false);
}

TEST(PageMallocator, Simple) {
	constexpr size_t size = 1u << 12;
	constexpr size_t page_alignment = 1u << 10;

	PageMallocator<page_alignment, size> s;
	Blk b = s.allocate();

	ASSERT_EQ(b.valid(), true);
	auto *ptr = (uint8_t *) b.ptr;
	for (size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}
	ASSERT_EQ(s.owns(b), true);
	s.deallocate(b);
}

TEST(FreeListPageMallocator, Simple) {
	constexpr size_t size = 1u << 12;
	constexpr size_t page_alignment = 1u << 10;

	FreeListPageMallocator<page_alignment, size> s;
	Blk b = s.allocate();

	ASSERT_EQ(b.valid(), true);
	auto *ptr = (uint8_t *) b.ptr;
	for (size_t i = 0; i < size; i++) {
		ptr[i] = i;
	}
	ASSERT_EQ(s.owns(b), true);
	s.deallocate(b);
}

TEST(STDAllocatorWrapper, simple) {
	constexpr size_t size = 1u << 4u;

	using T = uint64_t;
	using Allocator = StackAllocator<size>;
	using WrapperAllocator = STDAllocatorWrapper<T, Allocator>;
	WrapperAllocator s;

	const T *ret = WrapperAllocator::allocate(s, size);
	ASSERT_NE(ret, nullptr);

	const T *ret1 = WrapperAllocator::allocate(s, size+1);
	ASSERT_EQ(ret1, nullptr);

	using CV = std::vector<T, WrapperAllocator>;
	CV v = {0, 1, 2, 3};
	for(uint32_t i = 0; i < 4; i++) {
		ASSERT_EQ(v[i], i);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
