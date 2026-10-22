#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>


#include "algorithm/sat/bruteforce.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


template <typename T>
class SAT : public testing::Test {};

TYPED_TEST_SUITE_P(SAT);

TYPED_TEST_P(SAT, bruteforce) {
    constexpr static uint32_t n = 3;
    constexpr static uint32_t c = 6;

    uint32_t clauses[c] = {0b101, 0b110, 0b111, 0b100, 0b010, 0b000};
    auto sol = sat_bruteforce<TypeParam, n>(clauses, c);

    // reference: an assignment x satisfies clause j if some variable has
    // the polarity of the clause, i.e. (c_j & x) | (~c_j & ~x) != 0
    std::vector<TypeParam> e;
    for (uint32_t x = 0; x < (1u << n); ++x) {
        bool all = true;
        for (uint32_t j = 0; j < c; ++j) {
            all &= (((clauses[j] & x) | (~clauses[j] & ~x & ((1u << n) - 1u))) != 0);
        }
        if (all) { e.push_back(TypeParam(x)); }
    }
    std::sort(sol.begin(), sol.end());
    EXPECT_EQ(sol, e);
    EXPECT_EQ((sat_bruteforce<TypeParam, n, false>(clauses, c).size()), e.size());
}

REGISTER_TYPED_TEST_SUITE_P(SAT, bruteforce);
using MyTypes = ::testing::Types<uint32_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, SAT, MyTypes);


/// next_gray walks the Gray code: next_gray(to_gray(i)) == to_gray(i+1), and
/// consecutive codes differ in exactly one bit
TEST(sat, next_gray) {
	uint32_t g = 0;
	for (uint32_t i = 0; i < (1u << 16u); ++i) {
		ASSERT_EQ(g, cryptanalysislib::to_gray(i)) << i;
		const uint32_t n = cryptanalysislib::next_gray(g);
		ASSERT_EQ(cryptanalysislib::popcount::popcount(n ^ g), 1u) << i;
		g = n;
	}
	static_assert(cryptanalysislib::next_gray<uint8_t>(0) == 1 && cryptanalysislib::next_gray<uint8_t>(1) == 3 && cryptanalysislib::next_gray<uint8_t>(3) == 2);
}

/// SIMD version against a direct evaluation, random sets of 8 clauses
template<const uint32_t n>
static void check_bruteforce_simd() {
	for (uint32_t r = 0; r < 200; ++r) {
		uint32_t clauses[8];
		for (auto &c : clauses) { c = uint32_t(rand()) & ((1u << n) - 1u); }
		auto sol = sat_bruteforce<uint32x8_t, uint32_t, n, false>(clauses, 8);

		std::vector<uint32_t> e;
		for (uint32_t x = 0; x < (1u << n); ++x) {
			bool all = true;
			for (uint32_t j = 0; j < 8; ++j) {
				all &= (((clauses[j] & x) | (~clauses[j] & ~x & ((1u << n) - 1u))) != 0);
			}
			if (all) { e.push_back(x); }
		}
		std::sort(sol.begin(), sol.end());
		EXPECT_EQ(sol, e) << r;
	}
}

// NOTE: n < 4: a single (partial) block, n > 4: several unrolled blocks
TEST(sat, bruteforce_simd) {
	check_bruteforce_simd<3>();
	check_bruteforce_simd<6>();
	check_bruteforce_simd<11>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
