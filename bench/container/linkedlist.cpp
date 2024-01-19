#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"
#include "container/linkedlist/linkedlist.h"

constexpr uint64_t N = 1000000;
constexpr uint64_t THREADS = 6;

struct TestStruct {
public:
	uint64_t data;
	bool operator==(const TestStruct& b) const {
		return data == b.data;
	}
	std::strong_ordering operator<=>(const TestStruct& b) const {
		return data <=> b.data;
	}
	friend std::ostream& operator<<(std::ostream& os, TestStruct const &tc) {
		return os << tc.data;
	}
};

B63_BASELINE(Base, nn) {
	FreeList<TestStruct> linked_list = FreeList<TestStruct>();
	std::vector<std::thread> pool(THREADS);
	uint64_t ret = 0;

	for (size_t iters = 0; iters < nn; iters++) {
		for (size_t i = 0; i < THREADS; ++i) {
			pool[i] = std::thread([&i, &linked_list]() {
				TestStruct t;
				for (size_t j = 0; j < N; ++j) {
					t.data = j + i * N;
					linked_list.insert(t);
				}
			});
		}

		for (auto &t: pool) {
			t.join();
		}

		for (size_t i = 0; i < THREADS; ++i) {
			pool[i] = std::thread([&i, &linked_list, &ret]() {
				TestStruct t;
				for (size_t j = 0; j < N; ++j) {
					t.data = j + i * N;
					ret += linked_list.remove(t);
				}
			});
		}

		for (auto &t: pool) {
			t.join();
		}
	}


	B63_KEEP(ret);
}


//B63_BENCHMARK(BaseAndInternalArray, nn) {
//
//}

int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles", argc, argv);
	return 0;
}