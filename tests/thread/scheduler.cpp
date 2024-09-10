#include <gtest/gtest.h>

#include "thread/scheduler.h"
#include "thread/thread2.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


#include <stdio.h>
static void parallel_task(void *pArg,
                          struct scheduler *s,
                          struct sched_task_partition p,
                          uint32_t thread_num) {
	(void )pArg;
	(void )s;
	(void )p;
	std::cout << thread_num << std::endl;
}

int main() {
	thread_start();
	return 1;
	constexpr size_t s = 10;
	void *memory;
	sched_size needed_memory;

	struct scheduler sched;

	// TODO merge those two functions
	scheduler_init(&sched, &needed_memory, SCHED_DEFAULT, nullptr);
	memory = calloc(needed_memory, 1);
	scheduler_start(&sched, memory);

	{
		sched_task tasks[s];
		for (size_t i = 0; i < s; ++i) {
			scheduler_add(&sched, &tasks[i], parallel_task, nullptr, 1, 1);
		}

		for (size_t i = 0; i < s; ++i) {
			scheduler_join(&sched, &tasks[i]);
		}
	}
	scheduler_stop(&sched, 1);
	free(memory);
}


