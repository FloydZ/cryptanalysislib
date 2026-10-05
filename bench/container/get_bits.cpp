#include <cstdint>

#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

B63_BASELINE(SoImplemented, nn) {
	BinaryVector<n> v1{};
	uint64_t k = 0, res=0;
	uint64_t i=0, j=0, r;
	for (; k < nn*100000; k++) {
		B63_SUSPEND {
			v1.random();
			i = rand() % (n-63);
			j = i + 62;
		}

		r = v1.get_bits(i, j);
		B63_SUSPEND {
			res += r;
		}
	}



	B63_KEEP(res);
}

B63_BENCHMARK(SoImplementedInline, nn) {
	using T = BinaryVector<n>;

	T v1{};
	typedef typename T::LimbType LimbType;

	LimbType lmask=0, rmask=0;
	int64_t lower_limb=0, higher_limb=0, shift=0;

	uint64_t k=0, res=0, r;
	uint64_t i=0, j=0;
	for (; k < nn*100000; k++) {
		B63_SUSPEND {
			v1.random();
			i = rand() % (n-63);
			j = i + 62;

			lmask = T::higher_mask(i);
			rmask = T::lower_mask2(j);
			lower_limb = i/T::limb_bits_width();
			higher_limb = (j-1)/T::limb_bits_width();
			shift = i % T::limb_bits_width();;
		}

		r = v1.get_bits(lower_limb, higher_limb, lmask, rmask, shift);
		B63_SUSPEND {
			res += r;
		}
	}

	B63_SUSPEND {
		res += 1;
	}

	B63_KEEP(res);
}

B63_BENCHMARK(Old, nn) {
	using T = BinaryVector<n>;

	T v1{};
	typedef typename T::LimbType LimbType;

	uint64_t k=0, res=0;
	uint64_t i=0, j=0;
	for (; k < nn*100000; k++) {
		B63_SUSPEND {
			v1.random();
			i = rand() % (n-63);
			j = i + 62;
		}


		const LimbType lmask = T::higher_mask(i);
		const LimbType rmask = T::lower_mask2(j);
		const int64_t lower_limb = i/T::limb_bits_width();
		const int64_t higher_limb = (j-1)/T::limb_bits_width();
		uint64_t r;

		const uint64_t shift = i%T::limb_bits_width();
		if (lower_limb == higher_limb) {
			r =  (v1.data()[lower_limb] & lmask & rmask) >> (shift);
		} else {
			const LimbType a = v1.data()[lower_limb] & lmask;
			const LimbType b = v1.data()[higher_limb] & rmask;

			auto c = (a >> shift);
			auto d = (b << (T::limb_bits_width() - shift));
			r = c ^d;
		}

		B63_SUSPEND {
			res += r;
		}
	}

	B63_SUSPEND {
		res += 1;
	}

	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
