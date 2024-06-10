#include <gtest/gtest.h>

#include "alloc/gc_simple.h"
#include "helper.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

// TODO more tests: https://github.com/mkirchner/gc/blob/7f6f17c8b3425df6cd27d6f9385265b23034a793/test/test_gc.c
TEST(AllocationMap, Simple) {
	auto* am = new AllocationMap(8, 16, 0.5, 0.2, 0.8);
	EXPECT_NE(am, nullptr);

	// aks for something which is not there
	int *f = (int *)malloc(sizeof(int));
	auto *a = am->get(f);
	EXPECT_EQ(a, nullptr);
}

TEST(GarbageCollector, Simple) {
	GarbageCollector gc;
	void *bos = __builtin_frame_address(0);
	gc.gc_start(bos);

	int *arrya = (int *)gc.gc_calloc(1024, sizeof(int));
	for (uint32_t i = 0; i < 1024; ++i) {
		arrya[i] = i;
	}
	gc.gc_stop();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
