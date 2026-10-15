#include <algorithm>
#include <cstdint>
#include <list>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/fill.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

template <typename T>
class Fill : public testing::Test {};

TYPED_TEST_SUITE_P(Fill);

TYPED_TEST_P(Fill, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    cryptanalysislib::fill(in.begin(), in.end(), 1);
    for(size_t i = 0; i < s; i++) {
    	EXPECT_EQ(in[i], 1);
    }
}

TYPED_TEST_P(Fill, multithreaded) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    
    cryptanalysislib::fill(par_if(true),in.begin(), in.end(), 1);
    for(size_t i = 0; i < s; i++) {
    	EXPECT_EQ(in[i], 1);
    }
}


REGISTER_TYPED_TEST_SUITE_P(Fill, simple, multithreaded);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Fill, MyTypes);

/// `fill_n`, sequential and parallel: fills exactly n elements, returns first + n
TEST(Fill, fill_n) {
	for (const size_t s : {size_t(0), size_t(1), size_t(100), size_t(1u << 21)}) {
		std::vector<uint32_t> v(s + 5, 1u), w(s + 5, 1u);
		EXPECT_EQ(cryptanalysislib::fill_n(v.begin(), s, 7u), v.begin() + s);
		EXPECT_EQ(cryptanalysislib::fill_n(par_if(true), w.begin(), s, 7u), w.begin() + s);
		for (size_t i = 0; i < v.size(); ++i) {
			ASSERT_EQ(v[i], i < s ? 7u : 1u) << i;
			ASSERT_EQ(w[i], i < s ? 7u : 1u) << i;
		}
	}
}

/// non-contiguous and empty ranges (`memset` wrote a list as an array)
TEST(Fill, list_and_empty) {
	std::list<uint32_t> l(10, 1u);
	cryptanalysislib::fill(l.begin(), l.end(), 7u);
	for (const uint32_t x : l) { EXPECT_EQ(x, 7u); }
	std::vector<uint32_t> e;
	cryptanalysislib::fill(e.begin(), e.end(), 7u);
	EXPECT_TRUE(e.empty());
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
