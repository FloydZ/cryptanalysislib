#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <numeric>
#include <vector>

#include "algorithm/rotate.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// every in-place rotation algorithm against std::rotate: all lengths 0..70
/// and every split point `left`
template<typename T, typename F>
static void check_all(F f, const char *name) {
	for (size_t n = 0; n <= 70; ++n) {
		for (size_t left = 0; left <= n; ++left) {
			std::vector<T> v(n), e(n);
			std::iota(v.begin(), v.end(), T(1));
			e = v;
			f(v.data(), left, n - left);
			std::rotate(e.begin(), e.begin() + left, e.end());
			ASSERT_EQ(v, e) << name << " n=" << n << " left=" << left;
		}
	}
}

TEST(rotate, algorithms_uint32) {
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { contrev_rotation(a, l, r); }, "contrev");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { trinity_rotation<uint32_t, 8>(a, l, r); }, "trinity<8>");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { trinity_rotation<uint32_t, 32>(a, l, r); }, "trinity<32>");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { griesmills_rotation(a, l, r); }, "griesmills");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { grail_rotation(a, l, r); }, "grail");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { piston_rotation(a, l, r); }, "piston");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { helix_rotation(a, l, r); }, "helix");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { drill_rotation(a, l, r); }, "drill");
}

TEST(rotate, algorithms_uint8_uint64) {
	check_all<uint8_t>([](uint8_t *a, size_t l, size_t r) { helix_rotation(a, l, r); }, "helix u8");
	check_all<uint64_t>([](uint64_t *a, size_t l, size_t r) { trinity_rotation<uint64_t, 16>(a, l, r); }, "trinity u64");
}

TEST(rotate, iterator_version) {
	for (size_t n = 0; n <= 40; ++n) {
		for (size_t left = 0; left <= n; ++left) {
			std::vector<int> v(n), e(n);
			std::iota(v.begin(), v.end(), 0);
			e = v;
			const auto r = cryptanalysislib::rotate(v.begin(), v.begin() + left, v.end());
			const auto re = std::rotate(e.begin(), e.begin() + left, e.end());
			ASSERT_EQ(v, e) << n << " " << left;
			ASSERT_EQ(r - v.begin(), re - e.begin()) << n << " " << left;
		}
	}
	std::list<int> l = {1, 2, 3, 4, 5};
	cryptanalysislib::rotate(l.begin(), std::next(l.begin(), 2), l.end());
	EXPECT_EQ(l, (std::list<int>{3, 4, 5, 1, 2}));

	std::vector<int> in = {1, 2, 3, 4}, out(4);
	cryptanalysislib::rotate_copy(in.begin(), in.begin() + 1, in.end(), out.begin());
	EXPECT_EQ(out, (std::vector<int>{2, 3, 4, 1}));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
