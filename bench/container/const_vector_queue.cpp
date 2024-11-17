#include <benchmark/benchmark.h>
#include <queue>

#include "container/vector_queue.h"

constexpr uint64_t LS = 1u << 16;

template<typename T>
void BM_ConstVectorQueue(benchmark::State& state) {
	using Q = ConstVectorQueue<T>;
	Q q;

    T c = 0;
    for (auto _ : state) {
		for (uint32_t i=0; i<state.range(0); i++) {
			q.push(i);
		}

		for (uint32_t i=0; i<state.range(0); i++) {
            benchmark::DoNotOptimize(c += q.front());
			q.pop();
		}
	}

	state.SetComplexityN(state.range(0));
}

template<typename T>
void BM_Queue(benchmark::State& state) {
	using Q = std::queue<T>;
	Q q;

    T c = 0;
    for (auto _ : state) {
		for (uint32_t i=0; i<state.range(0); i++) {
			q.push(i);
		}

		for (uint32_t i=0; i<state.range(0); i++) {
            benchmark::DoNotOptimize(c += q.front());
			q.pop();
		}
	}

	state.SetComplexityN(state.range(0));
}


template<typename T>
void BM_ConstVectorQueue1(benchmark::State& state) {
	using Q = ConstVectorQueue<T>;
	Q q;

    T c = 0;
    for (auto _ : state) {
		for (uint32_t i=0; i<state.range(0); i++) {
			q.push(i);
            benchmark::DoNotOptimize(c += q.front());
			q.pop();
		}
	}

	state.SetComplexityN(state.range(0));
}

template<typename T>
void BM_Queue1(benchmark::State& state) {
	using Q = std::queue<T>;
	Q q;

    T c = 0;
    for (auto _ : state) {
		for (uint32_t i=0; i<state.range(0); i++) {
			q.push(i);
            benchmark::DoNotOptimize(c += q.front());
			q.pop();
		}
	}

	state.SetComplexityN(state.range(0));
}



BENCHMARK(BM_ConstVectorQueue<uint32_t>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_ConstVectorQueue1<uint32_t>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_Queue<uint32_t>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_Queue1<uint32_t>)->RangeMultiplier(2)->Range(128, LS)->Complexity();

int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
