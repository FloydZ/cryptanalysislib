#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <vector>

#include "random.h"
#include "algorithm/reverse.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class Reverse : public testing::Test {};
TYPED_TEST_SUITE_P(Reverse);

/// reverse against std::reverse, all sizes around the SIMD block sizes
TYPED_TEST_P(Reverse, simple) {
	for (size_t n = 0; n <= 300; ++n) {
		std::vector<TypeParam> v(n);
		for (auto &x : v) { x = TypeParam(rng()); }
		auto e = v;
		cryptanalysislib::reverse(v.begin(), v.end());
		std::reverse(e.begin(), e.end());
		EXPECT_EQ(v, e) << n;
	}
}

TYPED_TEST_P(Reverse, multithreading) {
	for (const size_t n : {size_t(1u << 21), size_t((1u << 21) + 3)}) {
		std::vector<TypeParam> v(n);
		for (auto &x : v) { x = TypeParam(rng()); }
		auto e = v;
		cryptanalysislib::reverse(par_if(true), v.begin(), v.end());
		std::reverse(e.begin(), e.end());
		EXPECT_EQ(v, e) << n;
	}
}

REGISTER_TYPED_TEST_SUITE_P(Reverse, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int32_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Reverse, MyTypes);

TEST(reverse, non_contiguous_and_constexpr) {
	std::list<int> l = {1, 2, 3};
	cryptanalysislib::reverse(l.begin(), l.end());
	EXPECT_EQ(l, (std::list<int>{3, 2, 1}));

	constexpr auto r = []() { std::array<uint32_t, 5> a = {1, 2, 3, 4, 5}; cryptanalysislib::reverse(a.begin(), a.end()); return a; }();
	static_assert(r[0] == 5 && r[4] == 1);
}

TEST(reverse, reverse_copy) {
	const std::vector<uint32_t> v = {1, 2, 3, 4, 5};
	std::vector<uint32_t> o(6, 0);
	EXPECT_EQ(cryptanalysislib::reverse_copy(v.begin(), v.end(), o.begin()), o.begin() + 5);
	EXPECT_EQ(o, (std::vector<uint32_t>{5, 4, 3, 2, 1, 0}));
	EXPECT_EQ(cryptanalysislib::reverse_copy(v.begin(), v.begin(), o.begin()), o.begin());
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
