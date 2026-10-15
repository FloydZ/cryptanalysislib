#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <vector>

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

/// `copy_if` against `std::copy_if`, also with a `std::list` input
TEST(copy, copy_if) {
	for (size_t s = 0; s <= 200; ++s) {
		std::vector<uint32_t> in(s), out(s, 0xffu), e(s, 0xffu);
		for (size_t i = 0; i < s; ++i) { in[i] = uint32_t(i * 2654435761u); }
		const auto pred = [](const uint32_t x) { return (x % 3u) == 1u; };
		const auto r = cryptanalysislib::copy_if(in.begin(), in.end(), out.begin(), pred);
		const auto re = std::copy_if(in.begin(), in.end(), e.begin(), pred);
		EXPECT_EQ(r - out.begin(), re - e.begin()) << s;
		EXPECT_EQ(out, e) << s;
	}

	const std::list<int> l = {1, -2, 3, -4, 5};
	std::vector<int> out;
	cryptanalysislib::copy_if(l.begin(), l.end(), std::back_inserter(out), [](const int x) { return x > 0; });
	EXPECT_EQ(out, (std::vector<int>{1, 3, 5}));
}

/// `copy_backward` against `std::copy_backward`: overlapping shift to the
/// right (memmove path) and non trivially copyable elements (loop path)
TEST(copy, copy_backward) {
	for (size_t s = 0; s <= 100; ++s) {
		for (const size_t shift : {size_t(0), size_t(1), size_t(7), size_t(50)}) {
			std::vector<uint32_t> v(s + shift), e;
			for (size_t i = 0; i < v.size(); ++i) { v[i] = uint32_t(i + 1u); }
			e = v;
			const auto r = cryptanalysislib::copy_backward(v.begin(), v.begin() + s, v.begin() + s + shift);
			const auto re = std::copy_backward(e.begin(), e.begin() + s, e.begin() + s + shift);
			EXPECT_EQ(r - v.begin(), re - e.begin()) << s << " " << shift;
			EXPECT_EQ(v, e) << s << " " << shift;
		}
	}

	std::vector<std::string> v = {"a", "b", "c", "d", "", ""}, e = v;
	cryptanalysislib::copy_backward(v.begin(), v.begin() + 4, v.end());
	std::copy_backward(e.begin(), e.begin() + 4, e.end());
	EXPECT_EQ(v, e);

	std::list<int> l = {1, 2, 3}, d = {0, 0, 0, 0};
	const auto r = cryptanalysislib::copy_backward(l.begin(), l.end(), d.end());
	EXPECT_EQ(r, std::next(d.begin()));
	EXPECT_EQ(d, (std::list<int>{0, 1, 2, 3}));
}

/// parallel `copy_if` against `std::copy_if`: several threads (large inputs,
/// uneven chunk sizes, almost nothing/everything copied) and the sequential
/// fallback (small inputs)
TEST(copy, copy_if_multithreaded) {
	for (const size_t s : {size_t(0), size_t(100), size_t(1u << 20u), size_t((1u << 20u) + 12345u)}) {
		for (const uint32_t mod : {3u, 1u << 30u}) {
			std::vector<uint32_t> in(s), out(s, 0xffu), e(s, 0xffu);
			for (size_t i = 0; i < s; ++i) { in[i] = uint32_t(i * 2654435761u); }
			const auto pred = [mod](const uint32_t x) { return (x % mod) == 1u; };
			const auto r = cryptanalysislib::copy_if(par_if(true), in.begin(), in.end(), out.begin(), pred);
			const auto re = std::copy_if(in.begin(), in.end(), e.begin(), pred);
			EXPECT_EQ(r - out.begin(), re - e.begin()) << s << " " << mod;
			EXPECT_EQ(out, e) << s << " " << mod;

			std::vector<uint32_t> all(s);
			const auto ra = cryptanalysislib::copy_if(par_if(true), in.begin(), in.end(), all.begin(), [](const uint32_t) { return true; });
			EXPECT_EQ(ra, all.end());
			EXPECT_EQ(all, in);
		}
	}

	// sequential policy
	std::vector<int> v = {1, -2, 3}, o(3);
	EXPECT_EQ(cryptanalysislib::copy_if(par_if(false), v.begin(), v.end(), o.begin(), [](const int x) { return x > 0; }) - o.begin(), 2);
	EXPECT_EQ(o[1], 3);
}

/// both are usable in constant expressions
TEST(copy, constexpr_copy_if_backward) {
	constexpr auto r = []() {
		uint32_t a[6] = {1, 2, 3, 4, 5, 6}, b[6] = {};
		cryptanalysislib::copy_if(a, a + 6, b, [](const uint32_t x) { return x & 1u; });
		cryptanalysislib::copy_backward(a, a + 4, a + 6);
		return a[5] * 1000u + a[2] * 100u + b[2] * 10u + b[3];
	}();
	static_assert(r == 4150u);
	EXPECT_EQ(r, 4150u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
