#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <numeric>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/numeric/transform_exclusive_scan.h"
#include "algorithm/numeric/transform_inclusive_scan.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// against the `std::` versions (with `+`, which is associative, so the
/// unspecified order of the `std::` versions does not matter), also inplace
TEST(transform_scan, against_std) {
	const auto sq = [](const uint32_t x) { return x * x; };
	for (size_t n = 0; n <= 200; ++n) {
		std::vector<uint32_t> a(n), out(n), e(n);
		for (auto &x : a) { x = uint32_t(rng()); }

		auto r = cryptanalysislib::transform_inclusive_scan(a.begin(), a.end(), out.begin(), std::plus<>(), sq);
		std::transform_inclusive_scan(a.begin(), a.end(), e.begin(), std::plus<>(), sq);
		EXPECT_EQ(r, out.end());
		EXPECT_EQ(out, e) << n;

		r = cryptanalysislib::transform_inclusive_scan(a.begin(), a.end(), out.begin(), std::plus<>(), sq, uint32_t(9));
		std::transform_inclusive_scan(a.begin(), a.end(), e.begin(), std::plus<>(), sq, uint32_t(9));
		EXPECT_EQ(r, out.end());
		EXPECT_EQ(out, e) << n;

		r = cryptanalysislib::transform_exclusive_scan(a.begin(), a.end(), out.begin(), uint32_t(9), std::plus<>(), sq);
		std::transform_exclusive_scan(a.begin(), a.end(), e.begin(), uint32_t(9), std::plus<>(), sq);
		EXPECT_EQ(r, out.end());
		EXPECT_EQ(out, e) << n;

		auto b = a;
		cryptanalysislib::transform_exclusive_scan(b.begin(), b.end(), b.begin(), uint32_t(9), std::plus<>(), sq);
		EXPECT_EQ(b, e) << n;
		b = a;
		cryptanalysislib::transform_inclusive_scan(b.begin(), b.end(), b.begin(), std::plus<>(), sq);
		std::transform_inclusive_scan(a.begin(), a.end(), e.begin(), std::plus<>(), sq);
		EXPECT_EQ(b, e) << n;
	}
}

/// operator order (string concatenation), accumulator types, input iterators
TEST(transform_scan, order_and_types) {
	const std::list<int> l = {1, 2, 3};
	const auto str = [](const int x) { return std::to_string(x); };
	std::vector<std::string> o;
	cryptanalysislib::transform_inclusive_scan(l.begin(), l.end(), std::back_inserter(o), std::plus<>(), str,
	                                           std::string(">"));
	EXPECT_EQ(o, (std::vector<std::string>{">1", ">12", ">123"}));
	o.clear();
	cryptanalysislib::transform_exclusive_scan(l.begin(), l.end(), std::back_inserter(o), std::string(">"),
	                                           std::plus<>(), str);
	EXPECT_EQ(o, (std::vector<std::string>{">", ">1", ">12"}));

	// without `init` the accumulator has the input value type (uint8_t)
	const std::vector<uint8_t> b = {200, 100};
	std::vector<uint32_t> w(2);
	cryptanalysislib::transform_inclusive_scan(b.begin(), b.end(), w.begin(), std::plus<>(),
	                                           [](const uint8_t x) { return uint32_t(x); });
	EXPECT_EQ(w, (std::vector<uint32_t>{200, 44}));
}

/// affine map x -> a*x + b (mod 2^32): composition is associative but not
/// commutative, so the parallel scans must keep the order of the chunks
struct Aff {
	uint32_t a, b;
	bool operator==(const Aff &) const = default;
};

/// parallel transform scans against the sequential ones: associative non
/// commutative operator, `+` against the `std::` versions, uneven sizes and
/// the sequential fallback
TEST(transform_scan, multithreaded) {
	const auto compose = [](const Aff &f, const Aff &g) { return Aff{f.a * g.a, f.b * g.a + g.b}; };
	const auto to_aff = [](const uint32_t x) { return Aff{2u * x + 1u, x}; };
	const auto sq = [](const uint32_t x) { return x * x; };
	for (const size_t n : {size_t(0), size_t(1), size_t(100), size_t(1u << 20u), size_t((1u << 20u) + 77u)}) {
		std::vector<uint32_t> in(n);
		for (auto &x : in) { x = uint32_t(rng()); }

		std::vector<Aff> out(n), e(n);
		auto r = cryptanalysislib::transform_inclusive_scan(par_if(true), in.begin(), in.end(), out.begin(), compose, to_aff, Aff{3u, 5u});
		cryptanalysislib::transform_inclusive_scan(in.begin(), in.end(), e.begin(), compose, to_aff, Aff{3u, 5u});
		EXPECT_EQ(r, out.end());
		EXPECT_TRUE(out == e) << n;

		auto r2 = cryptanalysislib::transform_exclusive_scan(par_if(true), in.begin(), in.end(), out.begin(), Aff{3u, 5u}, compose, to_aff);
		cryptanalysislib::transform_exclusive_scan(in.begin(), in.end(), e.begin(), Aff{3u, 5u}, compose, to_aff);
		EXPECT_EQ(r2, out.end());
		EXPECT_TRUE(out == e) << n;

		std::vector<uint32_t> o(n), oe(n);
		auto r3 = cryptanalysislib::transform_inclusive_scan(par_if(true), in.begin(), in.end(), o.begin(), std::plus<>(), sq);
		std::transform_inclusive_scan(in.begin(), in.end(), oe.begin(), std::plus<>(), sq);
		EXPECT_EQ(r3, o.end());
		EXPECT_EQ(o, oe) << n;

		cryptanalysislib::transform_inclusive_scan(par_if(true), in.begin(), in.end(), o.begin(), std::plus<>(), sq, uint32_t(9));
		std::transform_inclusive_scan(in.begin(), in.end(), oe.begin(), std::plus<>(), sq, uint32_t(9));
		EXPECT_EQ(o, oe) << n;

		cryptanalysislib::transform_exclusive_scan(par_if(true), in.begin(), in.end(), o.begin(), uint32_t(9), std::plus<>(), sq);
		std::transform_exclusive_scan(in.begin(), in.end(), oe.begin(), uint32_t(9), std::plus<>(), sq);
		EXPECT_EQ(o, oe) << n;
	}

	// without `init` the accumulator has the input value type (uint8_t)
	const size_t n = 1u << 20u;
	std::vector<uint8_t> b(n);
	for (auto &x : b) { x = uint8_t(rng()); }
	std::vector<uint32_t> w(n), we(n);
	const auto id = [](const uint8_t x) { return uint32_t(x); };
	cryptanalysislib::transform_inclusive_scan(par_if(true), b.begin(), b.end(), w.begin(), std::plus<>(), id);
	cryptanalysislib::transform_inclusive_scan(b.begin(), b.end(), we.begin(), std::plus<>(), id);
	EXPECT_EQ(w, we);
}

/// usable in constant expressions
TEST(transform_scan, constexpr_transform_scan) {
	constexpr auto r = []() {
		int a[3] = {1, 2, 3}, b[3] = {};
		const auto neg = [](const int x) { return -x; };
		cryptanalysislib::transform_exclusive_scan(a, a + 3, b, 10, std::plus<>(), neg);   // 10 9 7
		cryptanalysislib::transform_inclusive_scan(a, a + 3, a, std::plus<>(), neg);       // -1 -3 -6
		return b[0] * 10000 + b[1] * 1000 + b[2] * 100 - a[2];
	}();
	static_assert(r == 109706);
	EXPECT_EQ(r, 109706);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
