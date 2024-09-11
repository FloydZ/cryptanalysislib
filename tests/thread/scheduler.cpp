#include <gtest/gtest.h>

#include "thread/scheduler.h"
#include "thread/thread2.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


//static void parallel_task(void *pArg,
//                          struct scheduler *s,
//                          struct sched_task_partition p,
//                          uint32_t thread_num) {
//	(void )pArg;
//	(void )s;
//	(void )p;
//	std::cout << thread_num << std::endl;
//}
//
//int main() {
//	auto t = cryptanalysislib::thread{};
//	t.start();
//	return 1;
//
//	constexpr size_t s = 10;
//	void *memory;
//	sched_size needed_memory;
//
//	struct scheduler sched;
//
//	// TODO merge those two functions
//	scheduler_init(&sched, &needed_memory, SCHED_DEFAULT, nullptr);
//	memory = calloc(needed_memory, 1);
//	scheduler_start(&sched, memory);
//
//	{
//		sched_task tasks[s];
//		for (size_t i = 0; i < s; ++i) {
//			scheduler_add(&sched, &tasks[i], parallel_task, nullptr, 1, 1);
//		}
//
//		for (size_t i = 0; i < s; ++i) {
//			scheduler_join(&sched, &tasks[i]);
//		}
//	}
//	scheduler_stop(&sched, 1);
//	free(memory);
//}






/* Number of threads to start */
#define NTHREADS	100

using namespace cryptanalysislib;

/* This function will first increment count by 50, yield. When it gets the
 * control back, it will increment count again and then exit
 */
void *thread_func(void *arg) {
	int *count = (int *)arg;

	*count = *count + 50;
	printf("Thread %ld: Incremented count by 50 and will now yield\n", (unsigned long)mythread_self().tid);

	// mythread_yield();
	*count = *count + 50;
	printf("Thread %ld: Incremented count by 50 and will now exit\n", (unsigned long)mythread_self().tid);
	mythread_exit(nullptr);

	return nullptr;
}

/* This is a simple demonstration of how to use the mythread library.
 * Start NTRHEADS number of threads, collect count value and exit
 */
int main() {
	mythread_t threads[NTHREADS];
	int count[NTHREADS];
	char *status;

	for (uint32_t i = 0; i < NTHREADS; i++) {
		count[i] = i;
		mythread_create(&threads[i], nullptr, thread_func, &count[i]);
	}


	for (uint32_t i = 0; i < NTHREADS; i++) {
		printf("Main: Will now wait for thread %ld. Yielding..\n", (unsigned long)threads[i].tid);
		mythread_join(threads[i], (void **)&status);
		printf("Main: Thread %ld exited and increment count to %d\n", (unsigned long)threads[i].tid, count[i]);
	}

	printf("Main: All threads completed execution. Will now exit..\n");
	mythread_exit(nullptr);

	return 0;
}
