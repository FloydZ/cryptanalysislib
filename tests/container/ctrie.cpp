#include <gtest/gtest.h>
#include <iostream>

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
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
