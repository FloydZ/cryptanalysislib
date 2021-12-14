#include <iostream>
#include <vector>
// #include <omp.h>

#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"


/* Base Code
void parallelMalloc(int parallelism, int mallocCnt = 10000000) {
	omp_set_num_threads(parallelism);

	std::vector<char*> ptrStore(mallocCnt);

	boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();

	#pragma omp parallel for
	for (int i = 0; i < mallocCnt; i++) {
		ptrStore[i] = ((char*)malloc(100 * sizeof(char)));
	}

	boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();

	#pragma omp parallel for
	for (int i = 0; i < mallocCnt; i++) {
		free(ptrStore[i]);
	}

	boost::posix_time::ptime t3 = boost::posix_time::microsec_clock::local_time();


	boost::posix_time::time_duration malloc_time = t2 - t1;
	boost::posix_time::time_duration free_time   = t3 - t2;

	std::cout << " parallelism = "  << parallelism << "\t itr = " << mallocCnt <<  "\t malloc_time = " <<
	          malloc_time.total_milliseconds() << "\t free_time = " << free_time.total_milliseconds() << std::endl;
}*/


B63_BASELINE(Malloc, nn) {
	std::vector<char*> ptrStore(nn);

	int32_t res = 0;
	for (; res < nn; ++res) {
		ptrStore[res] = ((char*)malloc(100 * sizeof(char)));
	}

	B63_SUSPEND {
		res = 0;
		for (; res < nn; ++res) {
			free(ptrStore[res]);
		}
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}



B63_BENCHMARK(Free, nn) {
	std::vector<char*> ptrStore(nn);

	int32_t res = 0;


	B63_SUSPEND {
		for (; res < nn; ++res) {
			ptrStore[res] = ((char*)malloc(100 * sizeof(char)));
		}


	}

	res = 0;
	for (; res < nn; ++res) {
		free(ptrStore[res]);
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:cache-misses,lpe:cache-references,lpe:instructions", argc, argv);
	return 0;
}