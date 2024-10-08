#include "b63.h"
#include "counters/perf_events.h"

#include "container/hashmap.h"
#include "random.h"

using K = uint32_t;
using V = uint64_t;
constexpr uint32_t l = 24;
constexpr uint32_t bucketsize = 7;
constexpr uint32_t fillratio = 2;
V *data = nullptr;

constexpr static SimpleHashMapConfig s1 = SimpleHashMapConfig{bucketsize, 1u << l, 1};
constexpr static Simple2HashMapConfig s2 = Simple2HashMapConfig{1u << l};

using HM1 = SimpleHashMap<K, V, s1, Hash<K, 0, l, 2>>;
using HM2 = Simple2HashMap<K, V, s2, Hash<K, 0, l, 2>>;

HM1 *hm1;
HM2 *hm2;


B63_BENCHMARK(Simple, nn) {
	(void) b63run;
	uint64_t ret = 0;

	for (uint64_t s = 0; s < nn; ++s) {
		for (uint64_t i = 0; i < (1u << l) * fillratio; ++i) {
			hm1->insert(i, data[i]);
		}

		for (uint64_t i = 0; i < (1u << l) * fillratio; ++i) {
			for (uint64_t j = 0; j < hm1->load(i); ++j) {
				ret += hm1->ptr(j);
			}
		}

		hm1->clear();
	}

	B63_KEEP(ret);
}


B63_BASELINE(Simple2, nn) {
	(void) b63run;
	uint64_t ret = 0;

	for (uint64_t s = 0; s < nn; ++s) {
		for (uint64_t i = 0; i < (1u << l) * nn; ++i) {
			hm2->insert(i, data[i]);
		}

		for (uint64_t i = 0; i < (1u << l) * nn; ++i) {
			for (uint64_t j = 0; j < hm2->load(i); ++j) {
				ret += hm2->ptr(j);
			}
		}

		hm2->clear();
	}

	B63_KEEP(ret);
}

int main(int argc, char **argv) {
	srand(time(NULL));
	hm1 = new HM1;
	hm2 = new HM2;
	data = (V *) aligned_alloc(1024, (1u << l) * fillratio * sizeof(V));
	for (size_t i = 0; i < (1u << l) * fillratio; ++i) {
		data[i] = fastrandombytes_uint64();
	}

	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);

	delete hm1;
	delete hm2;
	free(data);
	return 0;
}
