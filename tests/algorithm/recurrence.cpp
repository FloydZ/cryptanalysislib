#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "algorithm/recurrence.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib;

/// computes s[0..len-1] from s[0..L-1] and the recurrence C
template<typename T>
static std::vector<T> expand(const std::vector<T> &init,
                             const std::vector<T> &C,
                             const size_t len,
                             const T mod) {
	std::vector<T> s(init);
	for (size_t i = init.size(); i < len; ++i) {
		T v = 0;
		for (size_t j = 1; j <= C.size(); ++j) {
			v = addmod<T>(v, mulmod<T>(C[j - 1], s[i - j], mod), mod);
		}
		s.push_back(v);
	}
	return s;
}

TEST(recurrence, fibonacci) {
	constexpr uint64_t mod = 1000000007;
	static_assert(linear_recurrence<uint64_t>({0, 1}, {1, 1}, 10, mod) == 55);
	const auto C = berlekamp_massey<uint64_t>({0, 1, 1, 2, 3, 5, 8, 13}, mod);
	EXPECT_EQ(C, (std::vector<uint64_t>{1, 1}));
	// F(1000) mod 1e9+7
	EXPECT_EQ(linear_recurrence<uint64_t>({0, 1}, C, 1000, mod), 517691607u);
}

template<typename T>
static void random_test(const T mod, const uint32_t max_L) {
	for (uint32_t it = 0; it < 200; ++it) {
		const uint32_t L = 1 + (uint32_t)(rng() % max_L);
		std::vector<T> C(L), init(L);
		for (uint32_t i = 0; i < L; ++i) {
			C[i] = (T)(rng() % mod);
			init[i] = (T)(rng() % mod);
		}
		C[L - 1] = 1u % mod; // full length recurrence
		const auto s = expand<T>(init, C, 2 * L + 50, mod);

		// BM recovers a recurrence that reproduces the whole sequence
		const std::vector<T> prefix(s.begin(), s.begin() + 2 * L);
		const auto C2 = berlekamp_massey<T>(prefix, mod);
		EXPECT_LE(C2.size(), L);
		const std::vector<T> init2(s.begin(), s.begin() + C2.size());
		EXPECT_EQ(expand<T>(init2, C2, s.size(), mod), s);

		// lin_rec computes every element
		for (size_t k = 0; k < s.size(); ++k) {
			EXPECT_EQ(linear_recurrence<T>(init, C, k, mod), s[k]);
		}
	}
}

TEST(recurrence, random_small_prime) {
	random_test<uint32_t>(10007u, 10);
}

TEST(recurrence, random_gf2) {
	// binary LFSRs
	random_test<uint32_t>(2u, 20);
}

TEST(recurrence, random_64bit_prime) {
	// 2**64 - 59: checks that nothing overflows
	random_test<uint64_t>(18446744073709551557ull, 8);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
