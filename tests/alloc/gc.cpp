#include <gtest/gtest.h>

#include "alloc/gc_simple.h"
#include "helper.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

// Source: https://github.com/mkirchner/gc/blob/7f6f17c8b3425df6cd27d6f9385265b23034a793/test/test_gc.c

static uint64_t dtor_ctr = 0;
void dtor(void* ptr) {
	(void)ptr;
	dtor_ctr += 1;
}


TEST(AllocationMap, Simple) {
	auto* am = new AllocationMap(8, 16, 0.5, 0.2, 0.8);
	EXPECT_NE(am, nullptr);

	// aks for something which is not there
	int *f = (int *)malloc(sizeof(int));
	auto *a = am->get(f);
	EXPECT_EQ(a, nullptr);

	free(f);
	delete am;
}

TEST(AllocationMap, new_delete) {
	auto* am = new AllocationMap(8, 16, 0.5, 0.2, 0.8);
	EXPECT_NE(am, nullptr);
	EXPECT_EQ(am->min_capacity, 11);
	EXPECT_EQ(am->capacity, 17);
	EXPECT_EQ(am->size, 0);
	EXPECT_EQ(am->sweep_limit, 8);
	EXPECT_EQ(am->downsize_factor, 0.2);
	EXPECT_EQ(am->upsize_factor, 0.8);
	EXPECT_NE(am->allocs, nullptr);

	delete am;
}

TEST(AllocationMap, basic_get) {
	auto* am = new AllocationMap(8, 16, 0.5, 0.2, 0.8);
	EXPECT_NE(am, nullptr);

	using T = int;

	T *f = (T *)malloc(sizeof(T));
	auto *a = am->get(f);
	EXPECT_EQ(a, nullptr);
	EXPECT_EQ(am->size, 0);

	*f = 5;
	a = am->put(f, sizeof(T));
	EXPECT_NE(a, nullptr);
	EXPECT_EQ(am->size, 1);
	EXPECT_NE(am->allocs, nullptr);

	const auto *b = am->get(f);
	EXPECT_EQ(a, b);
	EXPECT_EQ(a->ptr, b->ptr);
	EXPECT_EQ(a->ptr, f);

	a = am->put(f, sizeof(T));
	EXPECT_NE(a, nullptr);
	EXPECT_EQ(am->size, 1);
	EXPECT_NE(am->allocs, nullptr);

	a = am->put(f, sizeof(T));
	EXPECT_NE(a, nullptr);
	EXPECT_EQ(am->size, 1);
	EXPECT_NE(am->allocs, nullptr);

	am->remove(f, true);
	EXPECT_EQ(am->size, 0);
	auto *c = am->get(f);
	EXPECT_EQ(c, nullptr);

	free(f);
	delete am;
}

TEST(AllocationMap, put_get_remove) {
	using T = int;
	constexpr static size_t size = 64;
	T **d = (T **)malloc(size * sizeof(T *));
	for (uint32_t i = 0; i < size; ++i) { d[i] = (T *)malloc(sizeof(T)); }

	//
	auto *am = new AllocationMap(32, 32, DBL_MAX, 0, DBL_MAX);
	EXPECT_NE(am, nullptr);
	for (uint32_t i = 0; i < size; ++i) {
		am->put(d[i], sizeof(T));
	}
	EXPECT_EQ(am->size, size);

	for (uint32_t i = 0; i < size; ++i) {
		am->remove(d[i], true);
	}
	EXPECT_EQ(am->size, 0);

	for (uint32_t i = 0; i < size; ++i) { free(d[i]); }
	free(d);
	delete am;
}

TEST(GarbageCollector, Simple) {
	void *bos = __builtin_frame_address(0);
	GarbageCollector gc(bos);

	using T = uint32_t;
	constexpr size_t size = 1024;
	T *arrya = (T *) gc.calloc_ext(size, sizeof(T));
	for (uint32_t i = 0; i < size; ++i) {
		arrya[i] = i;
	}
}


TEST(GarbageCollector, Cleanup) {
	using T = int;
	constexpr static size_t size = 64;

	void *bos = __builtin_frame_address(0);
	GarbageCollector gc(bos, 32, 32, 0.0, DBL_MAX, DBL_MAX);

	T** ptrs = (T **)gc.malloc_(size * sizeof(T *));
	for (uint32_t i = 0; i < size; ++i) {
		ptrs[i] = nullptr;
	}

	for (uint32_t i = 0; i < 8; ++i) {
		for (uint32_t j = 0; j < size; ++j) {
			ptrs[j] = (T *) gc.malloc_((j+1) * sizeof(T));
		}

		for (uint32_t j = 0; j < size; ++j) {
			gc.free_(ptrs[j]);
		}
	}

	gc.free_(ptrs);

	for (uint32_t i = 0; i < gc.allocs->capacity; ++i) {
		EXPECT_EQ(gc.allocs->allocs[i], nullptr);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
