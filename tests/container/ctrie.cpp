#include <gtest/gtest.h>
#include <atomic>
#include <iostream>
#include <sstream>
#include <thread>

#include "container/ctrie.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using K = uint64_t;
using V = uint64_t;
using CTrie = CacheTrie<K, V>;
constexpr size_t limit = 1u << 20u;
constexpr size_t THREADS = 2;

uint64_t hash_(const uint64_t i) { return i; }

TEST(Ctrie, simple) {
	CTrie c;
	EXPECT_EQ(c.lookup(1ul), 0);
	c.insert(1, 1);
	EXPECT_EQ(c.lookup(1ul), 1);
	EXPECT_EQ(c.fast_lookup(1ul), 1);
}

TEST(Ctrie, insert) {
	CTrie c;
	for (size_t i = 0; i < limit; ++i) {
		c.insert(i, i);
	}
}

TEST(Ctrie, single_lookup) {
	CTrie c;
	for (size_t i = 0; i < limit; ++i) {
		c.insert(i, i);
	}

	EXPECT_EQ(c.lookup(1), 1);
}

TEST(Ctrie, lookup) {
	CTrie c;
	for (size_t i = 0; i < limit; ++i) {
		c.insert(i, i);
	}

	for (size_t i = 1; i < limit; ++i) {
		EXPECT_EQ(c.lookup(i), i);
	}

	for (size_t i = limit-1; i > 1; --i) {
		EXPECT_EQ(c.lookup(i), i);
	}

	for (size_t i = limit; i > 1; --i) {
		K key = rand()%limit;
		EXPECT_EQ(c.lookup(key), key);
	}

	EXPECT_EQ(c.lookup(limit), 0);
}

TEST(Ctrie, multithreaded_insert) {
	CTrie c{};
	std::vector<std::thread> pool(THREADS);

	for (size_t t = 0; t < THREADS; ++t) {
		pool[t] = std::thread([t, &c]() {
			for (size_t i = 0; i < limit; ++i) {
				c.insert(t*limit + i, t*limit + i);
			}
		});
	}

	for (auto &t: pool) {
		t.join();
	}

	for (size_t i = 1; i < THREADS*limit; ++i) {
		EXPECT_EQ(c.lookup(i), i);
	}
}

TEST(Ctrie, multithreaded_lookup) {
	CTrie c{};
	std::vector<std::thread> pool(THREADS);

	for (size_t i = 0; i < THREADS*limit; ++i) {
		c.insert(i, i);
	}
	for (size_t t = 0; t < THREADS; ++t) {
		pool[t] = std::thread([t, &c]() {
			for (size_t i = 0; i < limit; ++i) {
				EXPECT_EQ(c.lookup(t*limit + i), t*limit + i);
			}
		});
	}

	for (auto &t: pool) {
		t.join();
	}
}

/// overwrites, removes and concurrent inserts into expanded arrays.
/// NOTE: before, the count of an array created by an expansion stayed 0, a
/// 	remove wrapped it around, and the debug check `checkAANode` read past
/// 	the array.
TEST(Ctrie, overwrite_remove_reinsert) {
	CTrie c{};
	constexpr uint64_t N = 20000;
	for (uint64_t r = 0; r < 3; r++) {
		for (uint64_t i = 0; i < N; i++) { c.insert(i, i + r); }
	}
	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.lookup(i), i + 2); }
	for (uint64_t i = 0; i < N; i += 2) { c.remove(i); }
	for (uint64_t i = 1; i < N; i += 2) { EXPECT_EQ(c.lookup(i), i + 2); }

	std::vector<std::thread> pool(4);
	for (uint64_t t = 0; t < 4; t++) {
		pool[t] = std::thread([&c, t]() {
			for (uint64_t r = 0; r < 2; r++) {
				for (uint64_t i = 0; i < 5000; i++) { c.insert(N + t * 5000 + i, r); }
			}
		});
	}
	for (auto &t: pool) { t.join(); }
	for (uint64_t i = 0; i < 20000; i++) { EXPECT_EQ(c.lookup(N + i), 1u); }
}

TEST(Ctrie, fast_insert) {
	CTrie c;
	EXPECT_EQ(c.fast_lookup(1ul), 0);
	c.fast_insert(1, 1, hash_(1));
	EXPECT_EQ(c.fast_lookup(1ul), 1);
	EXPECT_EQ(c.fast_lookup(1ul), 1);
}

TEST(Ctrie, remove) {
	CTrie c;
	c.insert(1, 1);
	c.remove(1);
	EXPECT_EQ(c.lookup(1), 0);

	c.insert(2, 3);
	const V *v = c.fast_remove(2, hash_(2), nullptr, nullptr, 0);
	ASSERT_NE(v, nullptr);
	EXPECT_EQ(*v, 3);
	EXPECT_EQ(c.lookup(2), 0);
	EXPECT_EQ(c.fast_remove(2, hash_(2), nullptr, nullptr, 0), nullptr);
}

/// concurrent removes of everything (the arrays are compressed), then reinserts
TEST(Ctrie, multithreaded_remove_compress) {
	CTrie c{};
	constexpr uint64_t N = 1u << 16u;
	constexpr uint64_t T = 4;
	for (uint64_t i = 0; i < N; i++) { c.insert(i, i + 1); }

	std::vector<std::thread> pool(T);
	for (uint64_t t = 0; t < T; t++) {
		pool[t] = std::thread([&c, t]() {
			for (uint64_t i = t; i < N; i += T) {
				const V *v = c.remove(i);
				EXPECT_NE(v, nullptr) << i;
				if (v) { EXPECT_EQ(*v, i + 1); }
			}
		});
	}
	for (auto &t: pool) { t.join(); }
	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.lookup(i), 0u) << i; }

	for (uint64_t i = 0; i < N; i++) { c.insert(i, 2*i + 1); }
	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.lookup(i), 2*i + 1) << i; }
}

/// readers look up old keys, while writers insert new keys (expansions and
/// frozen slots on the lookup path)
TEST(Ctrie, multithreaded_lookup_during_insert) {
	CTrie c{};
	constexpr uint64_t N = 1u << 14u;
	for (uint64_t i = 0; i < N; i++) { c.insert(i, i + 1); }

	std::atomic<bool> done = false;
	std::vector<std::thread> readers(2), writers(4);
	for (uint64_t t = 0; t < writers.size(); t++) {
		writers[t] = std::thread([&c, t]() {
			for (uint64_t i = 0; i < (1u << 16u); i++) {
				const uint64_t k = N + t * (1u << 16u) + i;
				c.insert(k, k + 1);
			}
		});
	}
	for (auto &r : readers) {
		r = std::thread([&c, &done]() {
			while (!done.load()) {
				for (uint64_t i = 0; i < N; i += 7) { EXPECT_EQ(c.lookup(i), i + 1) << i; }
			}
		});
	}
	for (auto &t: writers) { t.join(); }
	done.store(true);
	for (auto &t: readers) { t.join(); }

	for (uint64_t k = 0; k < N + 4 * (1u << 16u); k++) { EXPECT_EQ(c.lookup(k), k + 1) << k; }
}

/// the cache is created by the inserts, moves to the level of the keys
/// (cache misses of the lookups), and the fast operations start there
TEST(Ctrie, cache) {
	CTrie c{};
	constexpr uint64_t N = 1u << 20u;
	EXPECT_EQ(c.cache_level(), -1);
	for (uint64_t i = 0; i < N; i++) { c.insert(i, i + 1); }
	EXPECT_GE(c.cache_level(), 8);

	// the keys are in the arrays at level 16
	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.lookup(i), i + 1) << i; }
	EXPECT_EQ(c.cache_level(), 16);

	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.fast_lookup(i), i + 1) << i; }
	EXPECT_EQ(c.fast_lookup(N), 0u);

	for (uint64_t i = 0; i < N; i += 3) { c.fast_insert(i, i + 2, i); }
	for (uint64_t i = N; i < N + 1000; i++) { c.fast_insert(i, i + 1, i); }
	for (uint64_t i = 0; i < N + 1000; i++) {
		EXPECT_EQ(c.fast_lookup(i), ((i < N) && (i % 3 == 0)) ? i + 2 : i + 1) << i;
	}

	for (uint64_t i = 1; i < N + 1000; i += 3) {
		const V *v = c.fast_remove(i);
		ASSERT_NE(v, nullptr) << i;
		EXPECT_EQ(*v, i + 1);
	}
	EXPECT_EQ(c.fast_remove(1), nullptr);
	for (uint64_t i = 0; i < N + 1000; i++) {
		const V e = (i % 3 == 1) ? 0 : (((i < N) && (i % 3 == 0)) ? i + 2 : i + 1);
		EXPECT_EQ(c.fast_lookup(i), e) << i;
		EXPECT_EQ(c.lookup(i), e) << i;
	}
}

/// concurrent fast inserts, lookups and removes on the cache
TEST(Ctrie, multithreaded_cache) {
	CTrie c{};
	constexpr uint64_t N = 1u << 16u;
	constexpr uint64_t T = 4;
	for (uint64_t i = 0; i < T * N; i++) { c.insert(i, i + 1); }
	for (uint64_t i = 0; i < T * N; i++) { EXPECT_EQ(c.lookup(i), i + 1); }
	EXPECT_GE(c.cache_level(), 8);

	std::vector<std::thread> pool(T);
	for (uint64_t t = 0; t < T; t++) {
		pool[t] = std::thread([&c, t]() {
			for (uint64_t i = t; i < T * N; i += T) {
				EXPECT_EQ(c.fast_lookup(i), i + 1) << i;
				if (i % 2) {
					const V *v = c.fast_remove(i);
					EXPECT_NE(v, nullptr) << i;
				} else {
					c.fast_insert(i, i + 2, i);
				}
				// new keys, expansions below the cached arrays
				c.fast_insert(T * N + i, i, T * N + i);
			}
		});
	}
	for (auto &t: pool) { t.join(); }

	for (uint64_t i = 0; i < T * N; i++) {
		EXPECT_EQ(c.lookup(i), (i % 2) ? 0 : i + 2) << i;
		EXPECT_EQ(c.fast_lookup(T * N + i), i) << i;
	}
}

/// without the cache the fast operations are the normal ones
TEST(Ctrie, no_cache) {
	constexpr static CacheTrieConfig config{{}, false};
	CacheTrie<K, V, std::hash<K>, std::equal_to<K>,
	          cryptanalysislib::hh::power_of_two_growth_policy<2>, config> c{};
	constexpr uint64_t N = 1u << 16u;
	for (uint64_t i = 0; i < N; i++) { c.fast_insert(i, i + 1, i); }
	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.fast_lookup(i), i + 1); }
	for (uint64_t i = 0; i < N; i += 2) { EXPECT_NE(c.fast_remove(i), nullptr); }
	for (uint64_t i = 0; i < N; i++) { EXPECT_EQ(c.lookup(i), (i % 2) ? i + 1 : 0); }
	EXPECT_EQ(c.cache_level(), -1);
}

TEST(Ctrie, print_dot) {
	CTrie c;
	for (uint64_t i = 0; i < 100; i++) { c.insert(i, i); }
	std::stringstream ss;
	c.print_dot(ss);
	const std::string s = ss.str();
	EXPECT_EQ(s.rfind("digraph ctrie {", 0), 0u);
	// every single node is printed
	size_t snodes = 0;
	for (size_t p = s.find("label=\"S "); p != std::string::npos; p = s.find("label=\"S ", p + 1)) { snodes++; }
	EXPECT_EQ(snodes, 100u);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
