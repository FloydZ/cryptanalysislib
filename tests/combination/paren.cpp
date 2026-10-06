#include <cstddef>
#include <vector>
#include <gtest/gtest.h>

#include "combination/paren.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// all words of n pairs of parenthesis in ascending (=colex) order
template<typename T>
std::vector<T> reference(const uint32_t n) {
	std::vector<T> ret;
	for (uint64_t x = 0; x < (1ull << (2*n)); x++) {
		if ((uint32_t)__builtin_popcountll(x) != n) { continue; }
		if (enumeration_parenthesis<T>::is_parenword(T(x))) { ret.push_back(T(x)); }
	}
	return ret;
}

template<typename T>
void check(const uint32_t n) {
	const auto ref = reference<T>(n);
	enumeration_parenthesis<T> b(n);
	EXPECT_EQ(ref.front(), enumeration_parenthesis<T>::first_parenword(n));
	EXPECT_EQ(ref.back(), enumeration_parenthesis<T>::last_parenword(n));
	for (const T r : ref) {
		EXPECT_EQ(b.next(), r);
	}
	// end of the sequence
	EXPECT_EQ(b.next(), T(0));
}

TEST(enum_paren, simple) {
	for (uint32_t n = 1; n <= 10; n++) {
		check<uint64_t>(n);
		check<uint32_t>(n);
	}
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
