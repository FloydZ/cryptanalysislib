#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <numeric>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/numeric/partial_sum.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class PartialSum : public testing::Test {};
TYPED_TEST_SUITE_P(PartialSum);

/// against `std::partial_sum`: every size around the SIMD block sizes,
/// out of place and inplace (the `uint32_t` SIMD path and the scalar one)
TYPED_TEST_P(PartialSum, all_sizes) {
	for (size_t n = 0; n <= 300; ++n) {
		std::vector<TypeParam> a(n), out(n + 1, TypeParam(7)), e(n + 1, TypeParam(7));
		for (auto &x : a) { x = TypeParam(rng()); }
		const auto r = cryptanalysislib::partial_sum(a.begin(), a.end(), out.begin());
		std::partial_sum(a.begin(), a.end(), e.begin());
		EXPECT_EQ(r, out.begin() + n);
		EXPECT_EQ(out, e) << n; // incl. the untouched last element

		cryptanalysislib::partial_sum(a.begin(), a.end(), a.begin());
		EXPECT_TRUE(std::equal(a.begin(), a.end(), e.begin())) << n;
	}
}

REGISTER_TYPED_TEST_SUITE_P(PartialSum, all_sizes);
using MyTypes = ::testing::Types<uint8_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, PartialSum, MyTypes);

/// operator order `op(acc, x)`, accumulator type, input iterators
TEST(partial_sum, operator_order) {
	const std::list<std::string> l = {"a", "b", "c"};
	std::vector<std::string> o;
	cryptanalysislib::partial_sum(l.begin(), l.end(), std::back_inserter(o));
	EXPECT_EQ(o, (std::vector<std::string>{"a", "ab", "abc"}));

	const std::vector<int> v = {5, 3, 1};
	std::vector<int> m(3);
	cryptanalysislib::partial_sum(v.begin(), v.end(), m.begin(), [](const int acc, const int x) { return acc * 10 + x; });
	EXPECT_EQ(m, (std::vector<int>{5, 53, 531}));

	// accumulator is uint8_t: wraps around although the output is wider
	const std::vector<uint8_t> b = {200, 100};
	std::vector<uint32_t> w(2);
	cryptanalysislib::partial_sum(b.begin(), b.end(), w.begin());
	EXPECT_EQ(w, (std::vector<uint32_t>{200, 44}));
}

/// parallel `partial_sum` against `std::partial_sum`: accumulator type,
/// custom `op` and the sequential fallback
TEST(partial_sum, multithreaded) {
	for (const size_t n : {size_t(0), size_t(1), size_t(100), size_t(1u << 20u), size_t((1u << 20u) + 31u)}) {
		std::vector<uint32_t> a(n), out(n + 1, 7u), e(n + 1, 7u);
		for (auto &x : a) { x = uint32_t(rng()); }
		const auto r = cryptanalysislib::partial_sum(par_if(true), a.begin(), a.end(), out.begin());
		std::partial_sum(a.begin(), a.end(), e.begin());
		EXPECT_EQ(r, out.begin() + n);
		EXPECT_EQ(out, e) << n; // incl. the untouched last element

		cryptanalysislib::partial_sum(par_if(true), a.begin(), a.end(), out.begin(), std::bit_xor<>());
		std::partial_sum(a.begin(), a.end(), e.begin(), std::bit_xor<>());
		EXPECT_EQ(out, e) << n;

		// accumulator is uint8_t: wraps around although the output is wider
		std::vector<uint8_t> b(n);
		std::vector<uint64_t> w(n), f(n);
		for (auto &x : b) { x = uint8_t(rng()); }
		cryptanalysislib::partial_sum(par_if(true), b.begin(), b.end(), w.begin());
		std::partial_sum(b.begin(), b.end(), f.begin());
		EXPECT_EQ(w, f) << n;
	}
}

/// usable in constant expressions (also the `uint32_t` overload)
TEST(partial_sum, constexpr_partial_sum) {
	constexpr auto r = []() {
		uint32_t a[4] = {1, 2, 3, 4};
		cryptanalysislib::partial_sum(a, a + 4, a);
		return a[0] * 1000u + a[1] * 100u + a[2] * 10u + a[3] % 10u;
	}();
	static_assert(r == 1360u);
	EXPECT_EQ(r, 1360u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
