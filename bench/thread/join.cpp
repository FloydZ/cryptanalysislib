#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "thread/thread2.h"
#include "pthread.h"

using namespace cryptanalysislib;
constexpr size_t size = 4;
size_t value = 0;
void *inc(void *a) noexcept {
	value += *((size_t *)a);
	return (void *)&value;
}

void *inc2(void *a) {
	value += *((size_t *)a);
	//mythread_yield();
	mythread_exit(nullptr);
	return nullptr;
}
void BM_pthread(benchmark::State& state) {
	std::vector<pthread_t> threads(state.range(0));
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			pthread_create(&threads[i], nullptr, inc, &i);
		}

		for (int64_t i = 0; i < state.range(0); ++i) {
			pthread_join(threads[i], nullptr);
		}
	}
}

void BM_thread(benchmark::State& state) {
	std::vector<mythread_t> threads(state.range(0));

	char *status = nullptr;
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			mythread_create(&threads[i], nullptr, inc2, &i);
		}

		for (int64_t i = 0; i < state.range(0); ++i) {
			mythread_join(threads[i], (void **)&status);
		}
	}
}


BENCHMARK(BM_pthread)->DenseRange(1, size,1);
BENCHMARK(BM_thread)->DenseRange(1, size,1);
int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return value;
}
