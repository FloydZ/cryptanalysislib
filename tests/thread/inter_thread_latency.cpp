#include <time.h>
#include <pthread.h>
#include <stdio.h>

typedef unsigned long long uint64;
const uint64 ITERATIONS = 500LL * 1000LL * 1000LL;

volatile uint64 s1 = 0;
volatile uint64 s2 = 0;

void* run(void*){
     uint64 value = s2;
    while (true) {
        while (value == s1) {
            // busy spin
        }
        value = __sync_add_and_fetch(&s2, 1);
    }
}

// Source:
//   https://mechanical-sympathy.blogspot.com/2011/08/inter-thread-latency.html
// Run:
//   taskset -c 2,4 ./tests/thread/test_thread_inter_thread_latency
int main (void) {
    pthread_t threads[1];
    pthread_create(&threads[0], NULL, run, NULL);

    timespec ts_start;
    timespec ts_finish;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

     uint64 value = s1;
    while (s1 < ITERATIONS) {
        while (s2 != value) {
            // busy spin
        }
        value = __sync_add_and_fetch(&s1, 1);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_finish);

    uint64 start = (ts_start.tv_sec * 1000000000LL) + ts_start.tv_nsec;
    uint64 finish = (ts_finish.tv_sec * 1000000000LL) + ts_finish.tv_nsec;
    uint64 duration = finish - start;

    printf("duration = %lld\n", duration);
    printf("ns per op = %lld\n", (duration / (ITERATIONS * 2)));
    printf("op/sec = %lld\n",  
        ((ITERATIONS * 2L * 1000L * 1000L * 1000L) / duration));
    printf("s1 = %lld, s2 = %lld\n", s1, s2);

    return 0;
}

