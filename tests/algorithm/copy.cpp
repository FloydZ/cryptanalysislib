#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/copy.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(copy, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);

    cryptanalysislib::copy(in.begin(), in.end(), out.begin());

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}


TEST(copy, multithreaded_single) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);


    cryptanalysislib::copy(par_if(false), in.begin(), in.end(), out.begin());

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}

TEST(copy, multithreaded_twothreads) {
    constexpr static size_t s = algorithmCopyConfig.min_size_per_thread*2;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);


    cryptanalysislib::copy(par_if(true), in.begin(), in.end(), out.begin());

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}


/// `copy_n`, sequential and parallel: copies exactly n elements, returns dest + n
TEST(copy, copy_n) {
	for (const size_t s : {size_t(0), size_t(1), size_t(100), size_t(1u << 21)}) {
		std::vector<uint64_t> in(s + 3), out(s + 3, 0xffu);
		for (size_t i = 0; i < in.size(); ++i) { in[i] = i * 7u + 1u; }
		auto r = cryptanalysislib::copy_n(in.begin(), s, out.begin());
		EXPECT_EQ(r, out.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(out[i], in[i]) << i; }
		for (size_t i = s; i < out.size(); ++i) { ASSERT_EQ(out[i], 0xffu) << i; }

		std::vector<uint64_t> out2(s + 3, 0xffu);
		auto r2 = cryptanalysislib::copy_n(par_if(true), in.begin(), s, out2.begin());
		EXPECT_EQ(r2, out2.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(out2[i], in[i]) << i; }
		for (size_t i = s; i < out2.size(); ++i) { ASSERT_EQ(out2[i], 0xffu) << i; }
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
