#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "thread/thread.h"

using namespace cryptanalysislib;

struct Node {
    int64_t value;
    Node *left  = nullptr;
    Node *right = nullptr;

    inline int64_t sum() noexcept {
        int64_t ret = value;
        if (left != nullptr) { ret += left->sum(); }
        if (right != nullptr) { ret += right->sum(); }
        return ret;
    }

    static Node* make_balanced_tree(const int64_t from,
                                    const int64_t to) noexcept {
        Node *ret = new Node;
        const int64_t value = from + (to - from)/2;
        ret->value = value;

        if (value > from) { make_balanced_tree(from, value-1ull); }
        if (value < to)   { make_balanced_tree(value+1ull, to); }
        return ret;
    }

    void free() noexcept {
        if (left != nullptr) { 
            left->free(); 
            delete left;
        }
        if (right != nullptr) { 
            right->free(); 
            delete right;
        }

        delete this;
    }
};

inline int64_t sum(Node *node) noexcept {
    return node->sum();
} 

struct SimpleSum {
    int64_t run(Node *node) noexcept {
        return node->sum();
    }
};


Node *root = nullptr;

struct SchedulerSum {
	StealingScheduler<> pool{};
    int64_t run(Node *node) noexcept {
        auto t = pool.enqueue(sum, node);
        return t.get();
    }
};


template<typename T>
static void BM_NodeSum(benchmark::State& state) {
    std::int64_t t = 0;
    T s;
	for ([[maybe_unused]] auto _ : state) {
	    t += s.run(root);
		benchmark::DoNotOptimize(t+=1);
	}
}

constexpr uint64_t limit = 1<<10;
BENCHMARK(BM_NodeSum<SimpleSum>);
BENCHMARK(BM_NodeSum<SchedulerSum>);


int main(int argc, char** argv) {
    root = Node::make_balanced_tree(0, 1000);
	
    ::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
    
    root->free();
	return 0;
}
