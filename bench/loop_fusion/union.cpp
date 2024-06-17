#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "loop_fusion/loop_fusion.hpp"

using namespace loop_fusion;

constexpr std::size_t limit = 1'000'000;
common::range r1(0, limit);
common::range r2(5'000, 400'000);
common::range r3(50'000, 600'000);
common::range r4(300'000, 900'000);

std::vector<int> a(limit);
std::vector<int> b(limit);

void generate() {
	std::random_device rd {};
	std::mt19937 gen { rd() };
	std::uniform_int_distribution dist { 1, 10'000 };

	auto generator = [&dist, &gen]() { return dist(gen); };
	std::generate(a.begin(), a.end(), generator);
	std::generate(b.begin(), b.end(), generator);
}

std::vector<int> c0(limit);
std::vector<int> d0(limit);
std::vector<int> c(limit);
std::vector<int> d(limit);

static void PlainLoop(benchmark::State& state) {
	for (auto _ : state) {
		for (std::size_t i = r1.start; i < r1.end; ++i) {
			// op1
			d0[i] = b[i];
			c0[i] = a[i] + b[i];
		}
		for (std::size_t i = r2.start; i < r2.end; ++i) {
			// op2
			d0[i] = c0[i] * a[i];
		}
		for (std::size_t i = r3.start; i < r3.end; ++i) {
			// op3
			c0[i] = d0[i] - a[i];
		}
		for (std::size_t i = r4.start; i < r4.end; ++i) {
			// op4
			d0[i] += a[i];
		}
	}
	state.SetComplexityN(state.range(0));
}

static void PlainLoopUnion(benchmark::State& state) {
	for (auto _ : state) {
	for (std::size_t i = r1.start; i < r2.start; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
	}
	for (std::size_t i = r2.start; i < r3.start; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
		// op2
		d[i] = c[i] * a[i];
	}
	for (std::size_t i = r3.start; i < r4.start; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
		// op2
		d[i] = c[i] * a[i];
		// op3
		c[i] = d[i] - a[i];
	}
	for (std::size_t i = r4.start; i < r2.end; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
		// op2
		d[i] = c[i] * a[i];
		// op3
		c[i] = d[i] - a[i];
		// op4
		d[i] += a[i];
	}
	for (std::size_t i = r2.end; i < r3.end; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
		// op3
		c[i] = d[i] - a[i];
		// op4
		d[i] += a[i];
	}
	for (std::size_t i = r3.end; i < r4.end; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
		// op4
		d[i] += a[i];
	}
	for (std::size_t i = r4.end; i < r1.end; ++i) {
		// op1
		d[i] = b[i];
		c[i] = a[i] + b[i];
	}
	}
}

static void Looper(benchmark::State& state) {
	for (auto _ : state) {
		auto op1 = [&](std::size_t i) {
			d[i] = b[i];
			c[i] = a[i] + b[i];
		};
		auto op2 = [&](std::size_t i) { d[i] = c[i] * a[i]; };
		auto op3 = [&](std::size_t i) { c[i] = d[i] - a[i]; };
		auto op4 = [&](std::size_t i) { d[i] += a[i]; };
		auto loop1 = main_range::loop(r1, op1);
		auto loop2 = main_range::loop(r2, op2);
		auto loop3 = main_range::loop(r3, op3);
		auto loop4 = main_range::loop(r4, op4);
		loop1.run();
		loop2.run();
		loop3.run();
		loop4.run();
	}
}
static void LooperCompileTimeUnion(benchmark::State& state) {
	for (auto _ : state) {
		auto op1 = [&](std::size_t i) {
			d[i] = b[i];
			c[i] = a[i] + b[i];
		};
		auto op2 = [&](std::size_t i) { d[i] = c[i] * a[i]; };
		auto op3 = [&](std::size_t i) { c[i] = d[i] - a[i]; };
		auto op4 = [&](std::size_t i) { d[i] += a[i]; };
		auto loop1 = compiletime::loop<0, 1'000'000>(op1);
		auto loop2 = compiletime::loop<5'000, 400'000>(op2);
		auto loop3 = compiletime::loop<50'000, 600'000>(op3);
		auto loop4 = compiletime::loop<300'000, 900'000>(op4);
		auto loop = loop1 | loop2 | loop3 | loop4;
		loop.run();
	}
}
static void LooperRunTimeUnion(benchmark::State& state) {
	for (auto _ : state) {
		auto op1 = [&](std::size_t i) {
			d[i] = b[i];
			c[i] = a[i] + b[i];
		};
		auto op2 = [&](std::size_t i) { d[i] = c[i] * a[i]; };
		auto op3 = [&](std::size_t i) { c[i] = d[i] - a[i]; };
		auto op4 = [&](std::size_t i) { d[i] += a[i]; };
		auto loop1 = runtime::looper(r1, op1);
		auto loop2 = runtime::looper(r2, op2);
		auto loop3 = runtime::looper(r3, op3);
		auto loop4 = runtime::looper(r4, op4);
		auto loop = loop1 | loop2 | loop3 | loop4;
		loop.run();
	}
}
static void LooperMainRangeUnion(benchmark::State& state) {
	for (auto _ : state) {
		auto op1 = [&](std::size_t i) {
			d[i] = b[i];
			c[i] = a[i] + b[i];
		};
		auto op2 = [&](std::size_t i) { d[i] = c[i] * a[i]; };
		auto op3 = [&](std::size_t i) { c[i] = d[i] - a[i]; };
		auto op4 = [&](std::size_t i) { d[i] += a[i]; };
		auto loop1 = main_range::loop(r1, op1);
		auto loop2 = main_range::loop(r2, op2);
		auto loop3 = main_range::loop(r3, op3);
		auto loop4 = main_range::loop(r4, op4);
		main_range::merge_and_run(loop1, loop2, loop3, loop4);
	}
}

BENCHMARK(PlainLoop)->Range(1024,1024);
BENCHMARK(PlainLoopUnion)->Range(1024,1024);
BENCHMARK(Looper)->Range(1024,1024);
BENCHMARK(LooperCompileTimeUnion)->Range(1024,1024);
BENCHMARK(LooperRunTimeUnion)->Range(1024,1024);
BENCHMARK(LooperMainRangeUnion)->Range(1024,1024);

int main(int argc, char** argv) {
	generate();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
