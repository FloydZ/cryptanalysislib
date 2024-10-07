#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "pthread.h"
#include "thread/thread.h"

constexpr size_t size = 8;
size_t value = 0;

#ifndef __APPLE__
using namespace cryptanalysislib;

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


static void stealingscheduler_create_destroy(benchmark::State& state) {
	for ([[maybe_unused]] auto _ : state) {
		StealingScheduler pool(state.range(0));
	}
}
BENCHMARK(stealingscheduler_create_destroy)->DenseRange(1, size, 1);

///**
// * Measure submitting a pre-packaged std::packaged_task.
// */
//static void submit_detach_packaged_task(benchmark::State& state) {
//	scheduler pool(state.range(0));
//	if (state.range(0)) {
//		pool.pause();
//	}
//
//	auto func = []{};
//
//	for ([[maybe_unused]] auto _ : state) {
//		std::packaged_task<void()> task(func);
//		pool.submit_detach(std::move(task));
//	}
//	pool.clear_task_queue();
//}
//BENCHMARK(submit_detach_packaged_task)->ArgName("paused")->Arg(true)->Arg(false);
//
///**
// * Measure submitting a lambda that returns void, not interested in a std::future.
// */
//static void submit_detach_void_lambda(benchmark::State& state) {
//	scheduler pool(size);
//	if (state.range(0)) {
//		pool.pause();
//	}
//
//	auto func = []{};
//
//	for ([[maybe_unused]] auto _ : state) {
//		pool.submit_detach(func);
//	}
//	pool.clear_task_queue();
//
//}
//BENCHMARK(submit_detach_void_lambda)->ArgName("paused")->Arg(true)->Arg(false);
//
///**
// * Measure submitting a lambda that returns void, with a std::future.
// */
//static void submit_void_lambda(benchmark::State& state) {
//	scheduler pool(size);
//	if (state.range(0)) {
//		pool.pause();
//	}
//
//	auto func = []{};
//
//	for ([[maybe_unused]] auto _ : state) {
//		auto f = pool.submit(func);
//		benchmark::DoNotOptimize(f);
//	}
//	pool.clear_task_queue();
//
//}
//BENCHMARK(submit_void_lambda)->ArgName("paused")->Arg(true)->Arg(false);
//
///**
// * Measure submitting a lambda that returns void, with std::future.
// */
//static void submit_void_lambda_future(benchmark::State& state) {
//	scheduler pool(NUM_THREADS);
//	if (state.range(0)) {
//		pool.pause();
//	}
//
//	auto func = []{};
//
//	for ([[maybe_unused]] auto _ : state) {
//		std::future<void> f = pool.submit(func);
//		benchmark::DoNotOptimize(f);
//	}
//
//	pool.clear_task_queue();
//}
//BENCHMARK(submit_void_lambda_future)->ArgName("paused")->Arg(true)->Arg(false);
//
//
///**
// * Measure submitting a lambda that returns int, with std::future.
// */
//static void submit_int_lambda_future(benchmark::State& state) {
//	scheduler pool(NUM_THREADS);
//	if (state.range(0)) {
//		pool.pause();
//	}
//
//	auto func = []{ return 1; };
//
//	for ([[maybe_unused]] auto _ : state) {
//		std::future<int> f = pool.submit(func);
//		benchmark::DoNotOptimize(f);
//	}
//
//	pool.clear_task_queue();
//}
//BENCHMARK(submit_int_lambda_future)->ArgName("paused")->Arg(true)->Arg(false);
//
///**
// * Measure running a pre-packaged std::packaged_task.
// */
//static void run_1k_packaged_tasks(benchmark::State& state) {
//	auto func = []{};
//
//	for ([[maybe_unused]] auto _ : state) {
//		scheduler pool(NUM_THREADS);
//		for (int i = 0; i < 1000; ++i) {
//			std::packaged_task<void()> task(func);
//			pool.submit_detach(std::move(task));
//		}
//	}
//}
//BENCHMARK(run_1k_packaged_tasks);
//
///**
// * Measure running a lot of lambdas.
// */
//static void run_1k_void_lambdas(benchmark::State& state) {
//	auto func = []{};
//
//	for ([[maybe_unused]] auto _ : state) {
//		scheduler pool(NUM_THREADS);
//		for (int i = 0; i < 1000; ++i) {
//			pool.submit_detach(func);
//		}
//	}
//}
//BENCHMARK(run_1k_void_lambdas);
//
///**
// * Measure running a lot of lambdas that return a value.
// */
//static void run_1k_int_lambdas(benchmark::State& state) {
//	auto func = []{ return 1; };
//
//	for ([[maybe_unused]] auto _ : state) {
//		scheduler pool(NUM_THREADS);
//		for (int i = 0; i < 1000; ++i) {
//			std::future<int> f = pool.submit(func);
//			benchmark::DoNotOptimize(f);
//		}
//	}
//}
//BENCHMARK(run_1k_int_lambdas);

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
	for (auto _ : state) {
		StealingScheduler local_pool(state.range(0));
		for (int64_t i = 0; i < state.range(0); ++i) {
			local_pool.enqueue_detach(inc, (void *)&i);
		}

		local_pool.wait_for_tasks();
	}
}

// NOTE: thread is just a wrapper pthread
//void BM_thread_pthread(benchmark::State& state) {
//	for (auto _ : state) {
//		cryptanalysislib::scheduler<pthread> local_pool(state.range(0));
//		for (int64_t i = 0; i < state.range(0); ++i) {
//			//local_pool.enqueue_detach(inc, (void *)&i);
//		}
//
//		//local_pool.wait_for_tasks();
//	}
//}

void BM_jthread(benchmark::State& state) {
	std::vector<std::jthread> threads(state.range(0));
	for (auto _ : state) {
		StealingScheduler local_pool(state.range(0));
		for (int64_t i = 0; i < state.range(0); ++i) {
			threads[i] = std::jthread(inc, (void *)&i);
		}

		for (int64_t i = 0; i < state.range(0); ++i) {
			threads[i].join();
		}
	}
}

void BM_stdthread(benchmark::State& state) {
	std::vector<std::thread> threads(state.range(0));
	for (auto _ : state) {
		StealingScheduler local_pool(state.range(0));
		for (int64_t i = 0; i < state.range(0); ++i) {
			threads[i] = std::thread(inc, (void *)&i);
		}

		for (int64_t i = 0; i < state.range(0); ++i) {
			threads[i].join();
		}
	}
}

BENCHMARK(BM_pthread)->DenseRange(1, size, 1);
BENCHMARK(BM_thread)->DenseRange(1, size, 1);
BENCHMARK(BM_jthread)->DenseRange(1, size, 1);
BENCHMARK(BM_stdthread)->DenseRange(1, size, 1);
//BENCHMARK(BM_thread_pthread)->DenseRange(1, size, 1);

#endif
int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return value;
}
