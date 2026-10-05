#include <gtest/gtest.h>
#include <iostream>

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
using BKT = BKTree<T>;

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

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
