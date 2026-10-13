#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <vector>

#include "random.h"
#include "algorithm/replace.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class Replace : public testing::Test {};
TYPED_TEST_SUITE_P(Replace);

/// replace against std::replace, all sizes around the vector width
TYPED_TEST_P(Replace, simple) {
	for (size_t n = 0; n <= 300; ++n) {
		std::vector<TypeParam> v(n);
		for (auto &x : v) { x = TypeParam(rng() % 4u); }
		auto e = v;
		cryptanalysislib::replace(v.begin(), v.end(), TypeParam(1), TypeParam(3));
		std::replace(e.begin(), e.end(), TypeParam(1), TypeParam(3));
		EXPECT_EQ(v, e) << n;
	}
}
REGISTER_TYPED_TEST_SUITE_P(Replace, simple);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int32_t, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Replace, MyTypes);

TEST(replace, non_contiguous) {
	std::list<int> l = {1, 2, 1, 3};
	cryptanalysislib::replace(l.begin(), l.end(), 1, 7);
	EXPECT_EQ(l, (std::list<int>{7, 2, 7, 3}));
}

TEST(replace, replace_if_copy_simd) {
	std::vector<uint32_t> v(1000), w, o(1000), e(1000);
	for (size_t i = 0; i < v.size(); ++i) { v[i] = (uint32_t)(i % 7u); }
	const auto odd = [](const uint32_t x) { return (x & 1u) != 0; };

	w = v; std::vector<uint32_t> we = v;
	cryptanalysislib::replace_if(w.begin(), w.end(), odd, 42u);
	std::replace_if(we.begin(), we.end(), odd, 42u);
	EXPECT_EQ(w, we);

	EXPECT_EQ(cryptanalysislib::replace_copy(v.begin(), v.end(), o.begin(), 3u, 9u), o.end());
	std::replace_copy(v.begin(), v.end(), e.begin(), 3u, 9u);
	EXPECT_EQ(o, e);

	EXPECT_EQ(cryptanalysislib::replace_copy_if(v.begin(), v.end(), o.begin(), odd, 8u), o.end());
	std::replace_copy_if(v.begin(), v.end(), e.begin(), odd, 8u);
	EXPECT_EQ(o, e);

	for (const size_t n : {size_t(0), size_t(1), size_t(31), size_t(1000)}) {
		w = v; we = v;
		cryptanalysislib::internal::replace_simd(w.data(), n, 2u, 5u);
		std::replace(we.begin(), we.begin() + n, 2u, 5u);
		EXPECT_EQ(w, we) << n;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
