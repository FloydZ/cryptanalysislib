#include <gtest/gtest.h>

#include "thread/thread.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(Thread, Simple) {
	//THREADS_PARALLEL(2)
	#pragma omp parallel
	{
		{
		printf("thread id: %d\n", Thread::get_tid());
		}
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
