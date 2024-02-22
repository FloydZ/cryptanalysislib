#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"
#include "container/fq_packed_vector.h"
#include "random.h"
#include <helper.h>

constexpr uint64_t ctr = 10;
using Row = kAryPackedContainer_T<uint64_t, 255, 3>;

B63_BASELINE(add, nn) {
	Row a, b, c;
	B63_SUSPEND {
		a.random();
		b.random();
	}

	for (uint32_t i = 0; i < ctr * nn; i++) {
		Row::add(c, a, b);
	}

	B63_KEEP(c.__data[0]);
}


B63_BENCHMARK(add_mod3_limb, nn) {
	uint64_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr * nn; i++) {
		c = Row::add_T(a, b);
		a = Row::add_T(b, c);
		b = Row::add_T(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(add_mod3_limb128, nn) {
	__uint128_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr * nn / 2; i++) {
		c = Row::add_T<__uint128_t>(a, b);
		a = Row::add_T<__uint128_t>(b, c);
		b = Row::add_T<__uint128_t>(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(add_mod3_limb256, nn) {
	uint64x4_t a, b, c;
	B63_SUSPEND {
		a = uint64x4_t::random();
		b = uint64x4_t::random();
	}

	for (uint32_t i = 0; i < ctr * nn / 4; i++) {
		c = Row::add256_T(a, b);
		a = Row::add256_T(b, c);
		b = Row::add256_T(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(add_mod3_limb256_nooverflow, nn) {
	uint64x4_t a, b, c;
	B63_SUSPEND {
		a = uint64x4_t::random();
		b = uint64x4_t::random();
	}

	for (uint32_t i = 0; i < ctr * nn / 4; i++) {
		c = Row::add256_T_no_overflow(a, b);
		a = Row::add256_T_no_overflow(b, c);
		b = Row::add256_T_no_overflow(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(sub_mod3_limb, nn) {
	uint64_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr * nn; i++) {
		c = Row::sub_T(a, b);
		a = Row::sub_T(b, c);
		b = Row::sub_T(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(sub_mod3_limb128, nn) {
	__uint128_t a, b, c;
	B63_SUSPEND {
		a = fastrandombytes_uint64();
		b = fastrandombytes_uint64();
	}

	for (uint32_t i = 0; i < ctr * nn / 2; i++) {
		c = Row::sub_T<__uint128_t>(a, b);
		a = Row::sub_T<__uint128_t>(b, c);
		b = Row::sub_T<__uint128_t>(a, c);
	}

	B63_KEEP(a);
}

B63_BENCHMARK(sub_mod3_limb256, nn) {
	uint64x4_t a, b, c;
	B63_SUSPEND {
		a = uint64x4_t::random();
		b = uint64x4_t::random();
	}

	for (uint32_t i = 0; i < ctr * nn / 4; i++) {
		c = Row::sub256_T(a, b);
		a = Row::sub256_T(b, c);
		b = Row::sub256_T(a, c);
	}

	B63_KEEP(a);
}


B63_BENCHMARK(hammingweight_mod3_limb, nn) {
	uint64_t a, weight = fastrandombytes_uint64();

	for (uint32_t i = 0; i < ctr * nn; i++) {
		B63_SUSPEND {
			a = fastrandombytes_uint64();
		}
		weight += Row::popcnt_T(a);
	}

	B63_KEEP(weight);
}

B63_BENCHMARK(hammingweight_mod3_limb128, nn) {
	__uint128_t a;
	uint64_t weight = fastrandombytes_uint64();
	B63_SUSPEND {
		a = fastrandombytes_uint64();
	}
	for (uint32_t i = 0; i < ctr * nn / 2; i++) {
		weight += Row::popcnt_T<__uint128_t>(a);
	}

	B63_KEEP(weight);
}

#ifdef USE_AVX2
B63_BENCHMARK(hammingweight_mod3_limb256, nn) {
	uint64x4_t a;
	uint64_t weight = 0;

	B63_SUSPEND {
		a = uint64x4_t::random();
	}

	for (uint32_t i = 0; i < ctr * nn / 4; i++) {
		weight += Row::popcnt256_T(a);
	}

	B63_KEEP(weight);
}
#endif


int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}