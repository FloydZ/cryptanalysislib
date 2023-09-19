#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"
#include <helper.h>
#include "random.h"
#include "container/fq_packed_vector.h"

constexpr uint64_t ctr = 10;
using Row = kAryPackedContainer_T<uint64_t, 255, 3>;

B63_BASELINE(add, nn) {
	Row a, b, c;
	B63_SUSPEND {
		a.random(); b.random();
	}

	for (uint32_t i = 0; i < ctr*nn; i++) {
		Row::add(c, a, b);
	}

	B63_KEEP(c.__data[0]);
}


B63_BENCHMARK(add_mod3_limb, nn) {
	uint64_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64(); b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr*nn; i++) {
		c = Row::add_mod3_limb(a, b);
		a = Row::add_mod3_limb(b, c);
		b = Row::add_mod3_limb(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(add_mod3_limb128, nn) {
	__uint128_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr*nn/2; i++) {
		c = Row::add_mod3_limb128(a, b);
		a = Row::add_mod3_limb128(b, c);
		b = Row::add_mod3_limb128(a, c);
	}

	B63_KEEP(a);
}

#ifdef USE_AVX2
B63_BENCHMARK(add_mod3_limb256, nn) {
	__m256i a, b, c;
	B63_SUSPEND {
		a = _mm256_set_epi64x(fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64());
		b = _mm256_set_epi64x(fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64());
	}

	for (uint32_t i = 0; i < ctr*nn/4; i++) {
		c = Row::add_mod3_limb256(a, b);
		a = Row::add_mod3_limb256(b, c);
		b = Row::add_mod3_limb256(a, c);
	}

	B63_KEEP(a);
}
#endif

B63_BENCHMARK(sub_mod3_limb, nn) {
	uint64_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr*nn; i++) {
		c = Row::sub_mod3_limb(a, b);
		a = Row::sub_mod3_limb(b, c);
		b = Row::sub_mod3_limb(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(sub_mod3_limb128, nn) {
	__uint128_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr*nn/2; i++) {
		c = Row::sub_mod3_limb128(a, b);
		a = Row::sub_mod3_limb128(b, c);
		b = Row::sub_mod3_limb128(a, c);
	}

	B63_KEEP(a);
}

#ifdef USE_AVX2
B63_BENCHMARK(sub_mod3_limb256, nn) {
	__m256i a, b, c;
	B63_SUSPEND {
		a = _mm256_set_epi64x(fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64());
		b = _mm256_set_epi64x(fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64());
	}

	for (uint32_t i = 0; i < ctr*nn/4; i++) {
		c = Row::sub_mod3_limb256(a, b);
		a = Row::sub_mod3_limb256(b, c);
		b = Row::sub_mod3_limb256(a, c);
	}

	B63_KEEP(a);
}
#endif


B63_BENCHMARK(hammingweight_mod3_limb, nn) {
	uint64_t a, weight = fastrandombytes_uint64();

	for (uint32_t i = 0; i < ctr*nn; i++) {
		B63_SUSPEND {
			a = fastrandombytes_uint64();
		}
		weight += Row::hammingweight_mod3_limb(a);
	}

	B63_KEEP(weight);
}

B63_BENCHMARK(hammingweight_mod3_limb128, nn) {
	__uint128_t a;
	uint64_t weight = fastrandombytes_uint64();
	B63_SUSPEND {
		a = fastrandombytes_uint64();
	}
	for (uint32_t i = 0; i < ctr*nn/2; i++) {


		weight += Row::hammingweight_mod3_limb128(a);
	}

	B63_KEEP(weight);
}

#ifdef USE_AVX2
B63_BENCHMARK(hammingweight_mod3_limb256, nn) {
	__m256i a;
	uint64_t weight = fastrandombytes_uint64();

	B63_SUSPEND {
		a = _mm256_set_epi64x(fastrandombytes_uint64(), fastrandombytes_uint64(), fastrandombytes_uint64(),fastrandombytes_uint64());
	}

	for (uint32_t i = 0; i < ctr*nn/4; i++) {
		weight += Row::hammingweight_mod3_limb256(a);
	}

	B63_KEEP(weight);
}
#endif


int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}