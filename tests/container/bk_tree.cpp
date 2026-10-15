#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "container/binary_packed_vector.h"
#include "container/bk_tree.h"
#include "helper.h"
#include "random.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using T = BinaryVector<100>;
using BKT = cryptanalysislib::BKTree<T>;

TEST(BKTree, first) {
	auto t = BKT {};

	for (size_t i = 0; i < 10000; ++i) {
		auto in = T{};
		in.random();
		t.insert(in);
	}

	uint32_t m = 100000;
	for (size_t i = 0; i < 10000; ++i) {
		auto in = T{};
		in.random();
		const uint32_t mn = t.lookup(in);
		if (mn < m) {m = mn; }
	}
	std::cout << m << std::endl;
}

/// lookup returns the minimal distance (compared with a brute force search)
TEST(BKTree, lookup_is_minimal) {
	using V = BinaryVector<64>;
	cryptanalysislib::BKTree<V> t;
	EXPECT_EQ(t.lookup(V{}), uint32_t(-1)); // empty
	std::vector<V> in(2000);
	for (auto &x : in) { x.random(); t.insert(x); }
	for (size_t r = 0; r < 200; ++r) {
		V q; q.random();
		uint32_t e = uint32_t(-1);
		for (const auto &x : in) { const uint32_t d = V::dist(q, x); if (d < e) { e = d; } }
		EXPECT_EQ(t.lookup(q), e) << r;
	}
	EXPECT_EQ(t.lookup(in[17]), 0u);
}

/// the concept rejects types without a `dist`
static_assert(cryptanalysislib::BKTreeMetric<BinaryVector<100>>);
static_assert(!cryptanalysislib::BKTreeMetric<int>);

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
