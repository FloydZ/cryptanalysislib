#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/shift.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// small threshold, so that the parallel code path is also taken for the
/// small and the non trivial inputs below
constexpr static AlgorithmShiftConfig small_config{{}, 1024};

/// sequential and parallel shift_left/shift_right against std:: for many
/// sizes and shift distances (also around the block size), the returned
/// iterator and the kept elements must match
template<typename T, typename G>
static void check(const size_t len, const ptrdiff_t n, G gen) {
	std::vector<T> v(len);
	for (size_t i = 0; i < len; ++i) { v[i] = gen(i); }

	for (const bool par : {false, true}) {
		auto a = v, e = v;
		auto r = par ? cryptanalysislib::shift_left<decltype(par_if(true)), decltype(a.begin()), small_config>(par_if(true), a.begin(), a.end(), n)
		             : cryptanalysislib::shift_left(a.begin(), a.end(), n);
		auto re = std::shift_left(e.begin(), e.end(), n);
		ASSERT_EQ(r - a.begin(), re - e.begin()) << "left " << len << " " << n << " " << par;
		ASSERT_TRUE(std::equal(a.begin(), r, e.begin())) << "left " << len << " " << n << " " << par;

		a = v; e = v;
		r = par ? cryptanalysislib::shift_right<decltype(par_if(true)), decltype(a.begin()), small_config>(par_if(true), a.begin(), a.end(), n)
		        : cryptanalysislib::shift_right(a.begin(), a.end(), n);
		re = std::shift_right(e.begin(), e.end(), n);
		ASSERT_EQ(r - a.begin(), re - e.begin()) << "right " << len << " " << n << " " << par;
		ASSERT_TRUE(std::equal(r, a.end(), re)) << "right " << len << " " << n << " " << par;
	}
}

TEST(shift, small) {
	for (size_t len = 0; len <= 40; ++len) {
		for (ptrdiff_t n = 0; n <= ptrdiff_t(len) + 2; ++n) {
			check<uint32_t>(len, n, [](size_t i) { return uint32_t(i * 7 + 1); });
		}
	}
}

TEST(shift, parallel) {
	const size_t len = (1u << 18) + 13;
	for (const ptrdiff_t n : {ptrdiff_t(1), ptrdiff_t(7), ptrdiff_t(1000), ptrdiff_t(1u << 16),
	                          ptrdiff_t(1u << 18) + 3, ptrdiff_t(len / 2), ptrdiff_t(len - 1)}) {
		check<uint64_t>(len, n, [](size_t i) { return uint64_t(i) * 0x9e3779b97f4a7c15ull; });
	}
}

/// non trivial type: the moved elements must survive (no self move, no
/// moved-from elements in the result)
TEST(shift, strings) {
	const size_t len = 1u << 16;
	for (const ptrdiff_t n : {ptrdiff_t(3), ptrdiff_t(5000)}) {
		check<std::string>(len, n, [](size_t i) { return std::string(20, char('a' + i % 26)) + std::to_string(i); });
	}
}

TEST(shift, forward_iterator) {
	std::list<int> l = {1, 2, 3, 4, 5};
	auto r = cryptanalysislib::shift_left(l.begin(), l.end(), 2);
	EXPECT_EQ(std::distance(l.begin(), r), 3);
	EXPECT_EQ(*l.begin(), 3);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
