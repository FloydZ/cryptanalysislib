#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "atomic_primitives.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Label_Type = uint64_t;

TEST(cmov, simple) {
	using T = uint32_t;
	T a = 1, b = 2;

	a = cmova<T>(a, b);
	EXPECT_EQ(a, b);
}


#ifdef __cpp_lib_atomic_wait
TEST(one_byte_mutex, single_threaded) {
	one_byte_mutex t;

	uint32_t ctr = 0;
	for (uint32_t i = 0; i < 1000; ++i) {
		t.lock();
		ctr += 1;
		t.unlock();
	}

	EXPECT_GE(ctr, 1000);
}

TEST(one_byte_mutex, mutli_threaded) {

	const uint32_t nr_threads = 2;
	uint32_t ctr = 0;
	one_byte_mutex t;

#pragma omp parallel default(none) shared(ctr, t) num_threads(nr_threads)
	{
#pragma omp parallel for
		for (uint32_t i = 0; i < 1000; ++i) {
			t.lock();
			ctr += 1;
			t.unlock();
		}
	}

	EXPECT_GE(ctr, 1000 * nr_threads);
}
#endif

TEST(FAA, simple) {
	const uint32_t nr_threads = 3;
	uint32_t ctr = 0;

#pragma omp parallel default(none) shared(ctr) num_threads(nr_threads)
	{
#pragma omp parallel for
		for (uint32_t i = 1; i < 1000; ++i) {
			FAA(&ctr, 2);
		}
	}

	EXPECT_GE(ctr, 1000 * nr_threads);
}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
