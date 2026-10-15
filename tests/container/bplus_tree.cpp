#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "random.h"
#include "container/bplus_tree.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// random inserts/erases/lookups against std::map
template<const uint32_t Order>
static void random_ops(const uint32_t ops, const uint64_t key_range, const bool verify_every_op) {
	bplus_tree<uint64_t, uint64_t, Order> t;
	std::map<uint64_t, uint64_t> m;
	for (uint32_t i = 0; i < ops; ++i) {
		const uint64_t k = rng() % key_range;
		const uint32_t op = rng() % 8;
		if (op < 3) {
			const bool e = m.insert({k, i}).second;
			ASSERT_EQ(t.insert(k, i), e);
		} else if (op == 3) {
			const bool e = m.insert_or_assign(k, i).second;
			ASSERT_EQ(t.insert_or_assign(k, i), e);
		} else if (op < 6) {
			ASSERT_EQ(t.erase(k), m.erase(k) == 1);
		} else {
			const auto it = m.find(k);
			const uint64_t *v = t.find(k);
			ASSERT_EQ(v != nullptr, it != m.end());
			if (v) { ASSERT_EQ(*v, it->second); }
		}

		ASSERT_EQ(t.size(), m.size());
		if (verify_every_op) { ASSERT_TRUE(t.verify()) << i; }
	}
	ASSERT_TRUE(t.verify());

	// in order iteration
	auto it = m.begin();
	for (auto [k, v] : t) {
		ASSERT_NE(it, m.end());
		ASSERT_EQ(k, it->first);
		ASSERT_EQ(v, it->second);
		++it;
	}
	ASSERT_EQ(it, m.end());

	// lower_bound
	for (uint32_t i = 0; i < 1000; ++i) {
		const uint64_t k = rng() % (key_range + 10);
		const auto e = m.lower_bound(k);
		const auto r = t.lower_bound(k);
		ASSERT_EQ(r == t.end(), e == m.end()) << k;
		if (e != m.end()) { ASSERT_EQ(r.key(), e->first); }
	}

	// erase everything
	for (const auto &[k, v] : m) {
		ASSERT_TRUE(t.erase(k));
	}
	ASSERT_TRUE(t.empty());
	ASSERT_TRUE(t.verify());
	ASSERT_EQ(t.begin(), t.end());
}

TEST(bplus_tree, random_order4) {
	random_ops<4>(20000, 300, true);
	random_ops<4>(100000, 1u << 20, false);
}

TEST(bplus_tree, random_order5) {
	random_ops<5>(20000, 300, true);
	random_ops<5>(100000, 1u << 20, false);
}

TEST(bplus_tree, random_order8) {
	random_ops<8>(20000, 1000, true);
	random_ops<8>(200000, 1u << 16, false);
}

TEST(bplus_tree, random_order64) {
	random_ops<64>(20000, 5000, true);
	random_ops<64>(200000, 1u << 16, false);
}

/// ascending and descending inserts, then erase from the front
TEST(bplus_tree, sequential) {
	bplus_tree<uint32_t, uint32_t, 6> t;
	for (uint32_t i = 0; i < 10000; ++i) { ASSERT_TRUE(t.insert(i, 2 * i)); }
	for (uint32_t i = 20000; i > 10000; --i) { ASSERT_TRUE(t.insert(i, 2 * i)); }
	ASSERT_TRUE(t.verify());
	EXPECT_EQ(t.size(), 20000u);
	EXPECT_GT(t.depth(), 3u);
	for (uint32_t i = 0; i < 10000; ++i) {
		ASSERT_TRUE(t.erase(i));
	}
	ASSERT_TRUE(t.verify());
	EXPECT_EQ(*t.find(15000), 30000u);
	EXPECT_EQ(t.find(5000), nullptr);
	EXPECT_EQ(t.begin().key(), 10001u);
}

TEST(bplus_tree, compare_and_strings) {
	bplus_tree<std::string, int, 4, std::greater<std::string>> t;
	const std::vector<std::string> w = {"delta", "alpha", "echo", "charlie", "bravo", "foxtrot"};
	for (size_t i = 0; i < w.size(); ++i) { EXPECT_TRUE(t.insert(w[i], int(i))); }
	EXPECT_FALSE(t.insert("alpha", 42));
	EXPECT_EQ(*t.find("alpha"), 1);
	ASSERT_TRUE(t.verify());

	std::vector<std::string> order;
	for (auto [k, v] : t) { order.push_back(k); }
	EXPECT_EQ(order, (std::vector<std::string>{"foxtrot", "echo", "delta", "charlie", "bravo", "alpha"}));

	bplus_tree<std::string, int, 4, std::greater<std::string>> u(std::move(t));
	EXPECT_TRUE(t.empty());
	EXPECT_EQ(u.size(), 6u);
	EXPECT_TRUE(u.contains("echo"));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
