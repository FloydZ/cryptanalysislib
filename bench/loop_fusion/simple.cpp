#include <benchmark/benchmark.h>
#include "loop_fusion/loop_fusion.hpp"

#include <algorithm>
#include <random>
using T = uint64_t;

template<typename T>
void func1(T *a, T *b, T *c, T *d, T *e, T *f, const size_t n) {
	for (size_t i = 0; i < n; ++i) {
		a[i] = b[i] + c[i];
	}

	//a[0] = e[1] + f[2];
	//a[1] = e[18] + f[13];
	//a[2] = e[1] + f[32];
	//a[3] = e[128] + f[678];
	//a[4] = e[1] + f[2];
	//a[5] = e[897] + f[1237];
	//a[1024] = e[1] + f[76];
	// TODO independe code
	for (size_t i = 0; i < n; ++i) {
		d[i] = a[i] * f[i] - b[i];
	}


	for (size_t i = 0; i < n; ++i) {
		a[i] = a[i] - f[i] - d[i];
	}
}
template<typename T>
void func4(T *a, T *b, T *c, T *d, T *e, T *f, const size_t n) {
	for (size_t i = 0; i < n; ++i) {
		a[i] = b[i] + c[i];
		d[i] = a[i] * f[i] - b[i];
		a[i] = a[i] - f[i] - d[i];
	}
}

template<typename T>
void func2(T *a, T *b, T *c, T *d, T *e, T *f, const size_t n) {
	using namespace loop_fusion::main_range;

	auto loop_body_1 = [&](size_t i) { a[i] = b[i] + c[i]; };
	auto loop_body_2 = [&](size_t i) { d[i] = a[i] * f[i] - b[i]; };
	auto loop_body_3 = [&](size_t i) { a[i] = a[i] - f[i] - d[i]; };

	// The following line merges and executes the two loop bodies
	// so that they are executed after another in just one loop.
	(loop_to(n) | loop_body_1 | loop_body_2 | loop_body_3).run();
}

template<typename T>
void func3(T *a, T *b, T *c, T *d, T *e, T *f, const size_t n) {
	using namespace loop_fusion::main_range;

	auto loop_body_1 = [&](size_t i) { a[i] = b[i] + c[i]; };
	auto loop_body_2 = [&](size_t i) { d[i] = a[i] * f[i] - b[i]; };
	auto loop_body_3 = [&](size_t i) { a[i] = a[i] - f[i] - d[i]; };

	(loop_to(n) | loop_body_1).run();
	(loop_to(n) | loop_body_2).run();
	(loop_to(n) | loop_body_3).run();
}

static void BM_Func1(benchmark::State &state) {
	const size_t n = state.range(0);
	T *a = new T[n];
	T *b = new T[n];
	T *c = new T[n];
	T *d = new T[n];
	T *e = new T[n];
	T *f = new T[n];

	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::uniform_int_distribution dist{1, 10'000};
	auto generator = [&dist, &gen]() { return dist(gen); };

	std::generate(a, a + n, generator);

	for (auto _: state) {
		// This code gets timed
		func1(a, b, c, d, e, f, state.range(0));
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] e;
	delete[] f;
}

static void BM_Func2(benchmark::State &state) {
	const size_t n = state.range(0);
	T *a = new T[n];
	T *b = new T[n];
	T *c = new T[n];
	T *d = new T[n];
	T *e = new T[n];
	T *f = new T[n];

	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::uniform_int_distribution dist{1, 10'000};
	auto generator = [&dist, &gen]() { return dist(gen); };

	std::generate(a, a + n, generator);

	for (auto _: state) {
		// This code gets timed
		func2(a, b, c, d, e, f, state.range(0));
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] e;
	delete[] f;
}

static void BM_Func3(benchmark::State &state) {
	const size_t n = state.range(0);
	T *a = new T[n];
	T *b = new T[n];
	T *c = new T[n];
	T *d = new T[n];
	T *e = new T[n];
	T *f = new T[n];

	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::uniform_int_distribution dist{1, 10'000};
	auto generator = [&dist, &gen]() { return dist(gen); };

	std::generate(a, a + n, generator);

	for (auto _: state) {
		// This code gets timed
		func3(a, b, c, d, e, f, state.range(0));
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] e;
	delete[] f;
}

static void BM_Func4(benchmark::State &state) {
	const size_t n = state.range(0);
	T *a = new T[n];
	T *b = new T[n];
	T *c = new T[n];
	T *d = new T[n];
	T *e = new T[n];
	T *f = new T[n];

	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::uniform_int_distribution dist{1, 10'000};
	auto generator = [&dist, &gen]() { return dist(gen); };

	std::generate(a, a + n, generator);

	for (auto _: state) {
		// This code gets timed
		func4(a, b, c, d, e, f, state.range(0));
	}

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] d;
	delete[] e;
	delete[] f;
}

// Register the function as a benchmark
BENCHMARK(BM_Func2)->Range(1ul << 10u, 1ul << 20u);
BENCHMARK(BM_Func1)->Range(1ul << 10u, 1ul << 20u);
BENCHMARK(BM_Func3)->Range(1ul << 10u, 1ul << 20u);
BENCHMARK(BM_Func4)->Range(1ul << 10u, 1ul << 20u);

// Run the benchmark
BENCHMARK_MAIN();
