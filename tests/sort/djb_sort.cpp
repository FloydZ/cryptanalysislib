#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "random.h"
#include "sort/djb_sort_portable.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// all pairs of 8 bit values, signed and unsigned
TEST(djbsort, minmax) {
	for (uint32_t i = 0; i < 256; ++i) {
		for (uint32_t j = 0; j < 256; ++j) {
			uint8_t a = uint8_t(i), b = uint8_t(j);
			djbsort::minmax(a, b);
			ASSERT_EQ(a, std::min(uint8_t(i), uint8_t(j)));
			ASSERT_EQ(b, std::max(uint8_t(i), uint8_t(j)));
			int8_t c = int8_t(i), d = int8_t(j);
			djbsort::minmax(c, d);
			ASSERT_EQ(c, std::min(int8_t(i), int8_t(j)));
			ASSERT_EQ(d, std::max(int8_t(i), int8_t(j)));
		}
	}

	// the extremes of the wide types
	uint64_t a = std::numeric_limits<uint64_t>::max(), b = 0;
	djbsort::minmax(a, b);
	EXPECT_EQ(a, 0u);
	int64_t c = std::numeric_limits<int64_t>::max(), d = std::numeric_limits<int64_t>::min();
	djbsort::minmax(c, d);
	EXPECT_EQ(c, std::numeric_limits<int64_t>::min());
	uint32_t e = 0xffffffffu, f = 0x7fffffffu;
	djbsort::minmax(e, f);
	EXPECT_EQ(e, 0x7fffffffu);
}

/// 0-1 principle: the network sorts all 0/1 inputs, so it sorts every input
TEST(djbsort, zero_one_principle) {
	for (uint32_t n = 0; n <= 16; ++n) {
		for (uint32_t m = 0; m < (1u << n); ++m) {
			std::vector<uint8_t> v(n);
			for (uint32_t i = 0; i < n; ++i) { v[i] = (m >> i) & 1u; }
			djbsort::sort(v.data(), n);
			ASSERT_TRUE(std::is_sorted(v.begin(), v.end())) << n << " " << m;
		}
	}
}

template <typename T>
class DjbSort : public testing::Test {};
TYPED_TEST_SUITE_P(DjbSort);

TYPED_TEST_P(DjbSort, random) {
	using T = TypeParam;
	std::vector<size_t> sizes;
	for (size_t n = 0; n <= 300; ++n) { sizes.push_back(n); }
	for (const size_t n : {1000ul, 4096ul, 10000ul, 65537ul}) { sizes.push_back(n); }

	for (const size_t n : sizes) {
		std::vector<T> v(n);
		for (auto &x : v) { x = T(rng()); }
		if (n > 2) {
			v[0] = std::numeric_limits<T>::min();
			v[n - 1] = std::numeric_limits<T>::max();
		}
		auto e = v, p = v;
		std::sort(e.begin(), e.end());
		djb_sort(v.data(), n);
		djbsort::sort(p.data(), int64_t(n));
		ASSERT_EQ(v, e) << n;
		ASSERT_EQ(p, e) << n;
	}
}

REGISTER_TYPED_TEST_SUITE_P(DjbSort, random);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, DjbSort, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
