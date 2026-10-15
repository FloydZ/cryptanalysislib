#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/transform.h"

#include "algorithm/accumulate.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;

TEST(transform, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);

    cryptanalysislib::transform(in.begin(), in.end(), out.begin(), [](T &a) {
    	return a;
    });

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}

TEST(transform_reduce, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto a =cryptanalysislib::transform_reduce(in1.begin(), in1.end(),
    									in2.begin(), 0,
    									std::plus<T>(),
    									std::multiplies<T>());
	EXPECT_EQ(a, cryptanalysislib::accumulate(in1.begin(), in1.end(), 0));
}



TEST(transform_reduce, multithreaded_single) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto a = cryptanalysislib::transform_reduce(par_if(false), in1.begin(), in1.end(), in2.begin(),
    									0,
    									std::plus<T>(),
    									std::multiplies<T>());
	EXPECT_EQ(a, cryptanalysislib::accumulate(in1.begin(), in1.end(), 0));
}

TEST(transform, multithreaded_twothreads) {
    constexpr static size_t s = 1000000;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);


    const auto a = cryptanalysislib::transform_reduce(par_if(false), in1.begin(), in1.end(), in2.begin(),
    									0,
    									std::plus<T>(),
    									std::multiplies<T>());
	EXPECT_EQ(a, cryptanalysislib::accumulate(in1.begin(), in1.end(), 0));
}


/// binary `transform` and all `transform_reduce` overloads against std::
TEST(Transform, binary_and_transform_reduce) {
	for (const size_t s : {size_t(0), size_t(1), size_t(13), size_t(1u << 21)}) {
		std::vector<uint64_t> a(s), b(s), o(s + 1, 0xdeadu), e(s);
		for (size_t i = 0; i < s; ++i) { a[i] = i & 0xffu; b[i] = (i * 7u) & 0xffu; }
		const auto add = [](uint64_t &x, uint64_t &y) { return x + y; };
		std::transform(a.begin(), a.end(), b.begin(), e.begin(), add);
		EXPECT_EQ(cryptanalysislib::transform(a.begin(), a.end(), b.begin(), o.begin(), add), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
		EXPECT_EQ(o[s], 0xdeadu);

		// inner product
		EXPECT_EQ(cryptanalysislib::transform_reduce(a.begin(), a.end(), b.begin(), uint64_t(3)),
		          std::transform_reduce(a.begin(), a.end(), b.begin(), uint64_t(3)));
		EXPECT_EQ(cryptanalysislib::transform_reduce(par_if(true), a.begin(), a.end(), b.begin(), uint64_t(3)),
		          std::transform_reduce(a.begin(), a.end(), b.begin(), uint64_t(3)));
		// reduce + unary transform
		const auto red = [](uint64_t x, uint64_t y) { return x + y; };
		const auto sq = [](uint64_t x) { return x * x; };
		EXPECT_EQ(cryptanalysislib::transform_reduce(a.begin(), a.end(), uint64_t(1), red, sq),
		          std::transform_reduce(a.begin(), a.end(), uint64_t(1), red, sq));
		EXPECT_EQ(cryptanalysislib::transform_reduce(par_if(true), a.begin(), a.end(), uint64_t(1), red, sq),
		          std::transform_reduce(a.begin(), a.end(), uint64_t(1), red, sq));
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
