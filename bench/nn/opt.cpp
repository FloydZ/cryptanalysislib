#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <cstdio>

//#define ENABLE_BENCHMARK
#include "nn/nn.h"

#ifndef BENCH_n
#define BENCH_n 256
#endif

#ifndef BENCH_LS
#define BENCH_LS 20
#endif

#ifndef BENCH_R
#define BENCH_R 8
#endif

#ifndef BENCH_K
#define BENCH_K ((BENCH_n) / (BENCH_R))
#endif

#ifndef BENCH_N
#define BENCH_N 150
#endif

#ifndef BENCH_GAMMA
#define BENCH_GAMMA 14
#endif

#ifndef BENCH_DELTA
#define BENCH_DELTA 10
#endif

#ifndef BENCH_BF
#define BENCH_BF 1000
#endif

#ifndef ITERS
#define ITERS 10
#endif


int main(int argc, char** argv) {
	random_seed(time(NULL));
	constexpr uint64_t LS = 1ul << BENCH_LS;
	constexpr static WindowedAVX2_Config config{BENCH_n, BENCH_R, BENCH_N, BENCH_K, LS, BENCH_DELTA, BENCH_GAMMA, 0, BENCH_BF};
	WindowedAVX2<config> algo{};
	config.print();

	constexpr bool solution = true;
	uint64_t time = 0;
	uint32_t sols = 0;
	for (size_t i = 0; i < ITERS; i++) {
		algo.generate_random_instance(solution);

		uint64_t t1 = clock();
		algo.avx2_nn(LS, LS);
		time += clock() - t1;
		sols += algo.solutions_nr;

		free(algo.L1);
		free(algo.L2);
	}

	algo.L1 = nullptr;
	algo.L2 = nullptr;

	double ctime = ((double)time/((double)ITERS))/CLOCKS_PER_SEC;
	printf("sols: %d\n", sols);
	printf("time: %f\n", ctime);
    return sols;
}
