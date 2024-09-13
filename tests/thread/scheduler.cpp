#include <gtest/gtest.h>

#include "thread/scheduler.h"
#include "thread/thread.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


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
	constexpr size_t s = 10;

	scheduler sched(SCHED_DEFAULT, nullptr);
	// NOTE: everything done in the constructor
	//sched.init(&needed_memory, SCHED_DEFAULT, nullptr);
	//memory = calloc(needed_memory, 1);
	//sched.start(memory);

	sched_task tasks[s];
	for (size_t i = 0; i < s; ++i) {
		sched.add(&tasks[i], parallel_task, nullptr, 1u<<10, 1<<9u);
	}

	for (size_t i = 0; i < s; ++i) {
		sched.join(&tasks[i]);
	}

	// NOTE: everything doen in the deconstructor
	// sched.stop(1u);
	// free(memory);
}
