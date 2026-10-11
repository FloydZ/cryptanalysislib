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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
