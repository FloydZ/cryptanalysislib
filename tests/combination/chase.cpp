#include <cstddef>
#include <bitset>
#include <set>
#include <gtest/gtest.h>

#include "combination/chase.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

uint64_t d = 0;
auto print_change = [](const uint16_t a,
                       const uint16_t b) {
	d ^= 1ull << a;
  	d ^= 1ull << b;
	std::bitset<10> dd(d);
	std::cout << dd << ":" << a << " " << b << std::endl;
};

TEST(BinaryChaseSequence, simple) {
	std::vector<std::pair<uint16_t, uint16_t>> ret{};
	constexpr uint32_t n = 4;
	constexpr uint32_t p = 1;
	BinaryChaseEnumerator<n, p>::changelist(ret);
	d = (1u << p) - 1u;

	std::bitset<n> dd(d);
	std::cout << dd << std::endl;
	for (const auto &l: ret) {
		d ^= 1ull << l.first;
		d ^= 1ull << l.second;
		EXPECT_EQ(cryptanalysislib::popcount::popcount(d), p);
		std::bitset<n> dd(d);
		std::cout << dd << ":" << l.first << ":" << l.second << std::endl;
	}
}

/// every change clears a set bit and sets a cleared one, at distance <= 2,
/// and all C(n, t) combinations are visited exactly once
template<const uint32_t n, const uint32_t t>
static void check_changelist() {
	std::vector<std::pair<uint16_t, uint16_t>> ret{};
	BinaryChaseEnumerator<n, t>::changelist(ret);
	ASSERT_EQ(ret.size(), bc(n, t) - 1u);

	uint64_t x = (1ull << t) - 1ull;
	std::set<uint64_t> seen{x};
	for (const auto &[a, b] : ret) {
		ASSERT_TRUE((x >> a) & 1u) << n << " " << t;
		ASSERT_FALSE((x >> b) & 1u) << n << " " << t;
		ASSERT_LE((a > b) ? a - b : b - a, 2) << n << " " << t;
		x ^= (1ull << a) ^ (1ull << b);
		ASSERT_LT(x, 1ull << n);
		seen.insert(x);
	}
	EXPECT_EQ(seen.size(), bc(n, t)) << n << " " << t;
}

TEST(BinaryChaseSequence, all_combinations) {
	check_changelist<5, 1>();
	check_changelist<5, 2>();
	check_changelist<6, 2>();
	check_changelist<8, 3>();
	check_changelist<10, 4>();
	check_changelist<12, 3>();
	check_changelist<12, 4>();
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
