#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "algorithm/move.h"

using ::testing::InitGoogleTest;

/// `move` against `std::move`: disjoint and overlapping (to the left) ranges
TEST(move, trivial) {
	for (size_t s = 0; s <= 100; ++s) {
		for (const size_t shift : {size_t(0), size_t(1), size_t(7), size_t(50)}) {
			std::vector<uint32_t> v(s + shift), e;
			for (size_t i = 0; i < v.size(); ++i) { v[i] = uint32_t(i + 1u); }
			e = v;
			const auto r = cryptanalysislib::move(v.begin() + shift, v.end(), v.begin());
			const auto re = std::move(e.begin() + shift, e.end(), e.begin());
			EXPECT_EQ(r - v.begin(), re - e.begin()) << s << " " << shift;
			EXPECT_EQ(v, e) << s << " " << shift;

			std::vector<uint32_t> out(s + shift, 0);
			EXPECT_EQ(cryptanalysislib::move(v.begin(), v.end(), out.begin()), out.end());
			EXPECT_EQ(out, v);
		}
	}
}

/// `move_backward` against `std::move_backward`: overlapping (to the right)
TEST(move, move_backward_trivial) {
	for (size_t s = 0; s <= 100; ++s) {
		for (const size_t shift : {size_t(0), size_t(1), size_t(7), size_t(50)}) {
			std::vector<uint64_t> v(s + shift), e;
			for (size_t i = 0; i < v.size(); ++i) { v[i] = uint64_t(i * 3u + 1u); }
			e = v;
			const auto r = cryptanalysislib::move_backward(v.begin(), v.begin() + s, v.end());
			const auto re = std::move_backward(e.begin(), e.begin() + s, e.end());
			EXPECT_EQ(r - v.begin(), re - e.begin()) << s << " " << shift;
			EXPECT_EQ(v, e) << s << " " << shift;
		}
	}
}

/// move only types: the elements are really moved
TEST(move, move_only) {
	std::vector<std::unique_ptr<int>> v, out(5);
	for (int i = 0; i < 5; ++i) { v.emplace_back(std::make_unique<int>(i)); }
	cryptanalysislib::move(v.begin(), v.end(), out.begin());
	for (int i = 0; i < 5; ++i) {
		EXPECT_EQ(v[i], nullptr);
		ASSERT_NE(out[i], nullptr);
		EXPECT_EQ(*out[i], i);
	}

	// shift right by two within the same range
	out.resize(7);
	cryptanalysislib::move_backward(out.begin(), out.begin() + 5, out.end());
	EXPECT_EQ(out[0], nullptr);
	EXPECT_EQ(out[1], nullptr);
	for (int i = 0; i < 5; ++i) { EXPECT_EQ(*out[i + 2], i); }

	std::list<std::string> l = {"a", "b", "c"};
	std::vector<std::string> d(3);
	cryptanalysislib::move(l.begin(), l.end(), d.begin());
	EXPECT_EQ(d, (std::vector<std::string>{"a", "b", "c"}));
	std::vector<std::string> d3;
	cryptanalysislib::move(d.begin(), d.end(), std::back_inserter(d3));
	EXPECT_EQ(d3, (std::vector<std::string>{"a", "b", "c"}));
	d = d3;

	std::list<std::string> d2(4);
	const auto r = cryptanalysislib::move_backward(d.begin(), d.end(), d2.end());
	EXPECT_EQ(r, std::next(d2.begin()));
	EXPECT_EQ(d2, (std::list<std::string>{"", "a", "b", "c"}));
}

/// unqualified calls are not ambiguous with `std::move`/`std::move_backward`
TEST(move, unqualified) {
	using namespace cryptanalysislib;
	std::vector<int> v = {1, 2, 3}, out(3), out2(3);
	move(v.begin(), v.end(), out.begin());
	move_backward(v.begin(), v.end(), out2.end());
	EXPECT_EQ(out, v);
	EXPECT_EQ(out2, v);
}

/// parallel `move`: several threads (trivial and non trivial elements, an
/// uneven size) and the sequential fallback (small inputs)
TEST(move, multithreaded) {
	for (const size_t s : {size_t(0), size_t(100), size_t(1u << 20u), size_t((1u << 20u) + 4321u)}) {
		std::vector<uint32_t> v(s), out(s + 1, 0xffu);
		for (size_t i = 0; i < s; ++i) { v[i] = uint32_t(i * 7u + 1u); }
		const auto r = cryptanalysislib::move(cryptanalysislib::par_if(true), v.begin(), v.end(), out.begin());
		EXPECT_EQ(r, out.begin() + s);
		EXPECT_TRUE(std::equal(v.begin(), v.end(), out.begin())) << s;
		EXPECT_EQ(out[s], 0xffu);
	}

	const size_t s = (1u << 18u) + 17u;
	std::vector<std::string> v(s), out(s);
	for (size_t i = 0; i < s; ++i) { v[i] = std::string(20, char('a' + (i % 26))) + std::to_string(i); }
	const auto e = v;
	EXPECT_EQ(cryptanalysislib::move(cryptanalysislib::par_if(true), v.begin(), v.end(), out.begin()), out.end());
	EXPECT_EQ(out, e);
	for (size_t i = 0; i < s; ++i) { ASSERT_TRUE(v[i].empty()) << i; } // moved from (libstdc++)

	std::vector<std::unique_ptr<int>> u, uo(3);
	for (int i = 0; i < 3; ++i) { u.emplace_back(std::make_unique<int>(i)); }
	cryptanalysislib::move(cryptanalysislib::par_if(false), u.begin(), u.end(), uo.begin());
	EXPECT_EQ(*uo[2], 2);
	EXPECT_EQ(u[2], nullptr);
}

/// both are usable in constant expressions
TEST(move, constexpr_move) {
	constexpr auto r = []() {
		uint32_t a[5] = {1, 2, 3, 4, 5};
		cryptanalysislib::move(a + 1, a + 5, a);         // 2 3 4 5 5
		cryptanalysislib::move_backward(a, a + 3, a + 5); // 2 3 2 3 4
		return a[0] * 10000u + a[1] * 1000u + a[2] * 100u + a[3] * 10u + a[4];
	}();
	static_assert(r == 23234u);
	EXPECT_EQ(r, 23234u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
