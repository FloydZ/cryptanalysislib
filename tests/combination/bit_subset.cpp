#include <cstddef>
#include <gtest/gtest.h>

#include "combination/bit_subset.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(bit_subset, p1) {
	using T = uint64_t;
	T V = 0b11010000100001;
	bit_subset_T<T> b(V);
	do {
		print_binary(b.current(), 14);
	} while (b.next());
}

/// `next` enumerates all non-empty subsets of V (ending with V), `first`/`last` reset
TEST(bit_subset, first_last) {
	using T = uint64_t;
	const T V = 0b11010000100001;
	bit_subset_T<T> b(V);
	uint64_t n = 0;
	while (b.next()) {
		EXPECT_EQ(b.current() & ~V, 0u);
		n += 1;
	}
	EXPECT_EQ(b.current(), V);
	// all non-empty subsets (the empty set is the start)
	EXPECT_EQ(n, (1ull << __builtin_popcountll(V)) - 1u);

	EXPECT_EQ(b.first(), 0u);
	EXPECT_EQ(b.current(), 0u);
	EXPECT_EQ(b.last(), V);
	EXPECT_EQ(b.first(0b101), 0u);
	EXPECT_EQ(b.full_set(), 0b101u);
	EXPECT_EQ(b.last(0b110), 0b110u);
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
