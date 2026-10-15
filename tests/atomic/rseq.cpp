#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "atomic/rseq.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// number of threads: more than cpus, so that threads get preempted and
/// migrated within the critical sections
static uint32_t nr_threads() {
	const uint32_t n = std::thread::hardware_concurrency();
	return 2u * (n == 0 ? 4u : n);
}

TEST(percpu, cpu) {
	std::cout << "rseq available: " << percpu::rseq_available() << std::endl;
	EXPECT_LT(percpu::cpu(), percpu::number_of_cpus());

#if defined(__linux__)
	// pinned to a cpu, `cpu()` must return it
	cpu_set_t old;
	ASSERT_EQ(sched_getaffinity(0, sizeof(old), &old), 0);
	for (uint32_t c = 0; c < percpu::number_of_cpus(); ++c) {
		if (!CPU_ISSET(c, &old)) { continue; }
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(c, &set);
		ASSERT_EQ(sched_setaffinity(0, sizeof(set), &set), 0);
		EXPECT_EQ(percpu::cpu(), c);
	}
	ASSERT_EQ(sched_setaffinity(0, sizeof(old), &old), 0);
#endif
}

TEST(percpu, counter) {
	constexpr uint64_t iters = 1u << 20;
	auto c = std::make_unique<percpu::counter<uint64_t>>();
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nr_threads(); ++t) {
		threads.emplace_back([&c, t]() {
			for (uint64_t i = 0; i < iters; ++i) {
				c->add(t + 1);
			}
		});
	}
	for (auto &t : threads) { t.join(); }

	const uint64_t n = nr_threads();
	EXPECT_EQ(c->sum(), iters * n * (n + 1) / 2);
}

/// too few shards: the shards are shared between cpus (atomics)
TEST(percpu, counter_shared_shards) {
	constexpr uint64_t iters = 1u << 18;
	percpu::counter<uint64_t, 1> c;
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nr_threads(); ++t) {
		threads.emplace_back([&c]() {
			for (uint64_t i = 0; i < iters; ++i) {
				c.add(1);
			}
		});
	}
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(c.sum(), iters * nr_threads());
}

struct node {
	node *next;
	uint64_t value;
};

/// every thread pushes its own nodes and pops/pushes random ones, at the
/// end every node must be in exactly one list
template<const uint32_t max_cpus>
static void percpu_list_test() {
	constexpr uint32_t nodes_per_thread = 1024;
	constexpr uint32_t iters = 1u << 18;
	const uint32_t n = nr_threads();
	auto l = std::make_unique<percpu::list<node, max_cpus>>();
	std::vector<node> nodes(n * nodes_per_thread);
	for (size_t i = 0; i < nodes.size(); ++i) { nodes[i].value = i; }

	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < n; ++t) {
		threads.emplace_back([&l, &nodes, t]() {
			for (uint32_t i = 0; i < nodes_per_thread; ++i) {
				l->push(&nodes[t * nodes_per_thread + i]);
			}
			for (uint32_t i = 0; i < iters; ++i) {
				if (node *e = l->pop()) {
					l->push(e);
				}
			}
		});
	}
	for (auto &t : threads) { t.join(); }

	std::vector<uint32_t> seen(nodes.size(), 0);
	l->clear([&seen](node *e) { seen[e->value] += 1; });
	for (size_t i = 0; i < seen.size(); ++i) {
		ASSERT_EQ(seen[i], 1u) << i;
	}
}

TEST(percpu, list) {
	percpu_list_test<1024>();
}

/// too few lists: the lists are shared between cpus (lock)
TEST(percpu, list_shared_lists) {
	percpu_list_test<1>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
