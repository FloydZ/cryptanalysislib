#include <gtest/gtest.h>
#include <algorithm>
#include <compare>
#include <cstdint>
#include <limits>
#include <list>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/lexicographical_compare.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class LexicographicalCompare : public testing::Test {};
TYPED_TEST_SUITE_P(LexicographicalCompare);

/// SIMD and scalar paths against the `std::` versions: equal prefixes of every
/// length, the first difference at every position, different lengths
TYPED_TEST_P(LexicographicalCompare, all_positions) {
	for (size_t n1 = 0; n1 <= 80; ++n1) {
		for (const size_t n2 : {size_t(0), n1 / 2, n1, n1 + 1, size_t(80)}) {
			std::vector<TypeParam> a(n1), b(n2);
			for (size_t i = 0; i < n1; ++i) { a[i] = TypeParam(rng()); }
			for (size_t i = 0; i < n2; ++i) { b[i] = (i < n1) ? a[i] : TypeParam(rng()); }
			for (size_t pos = 0; pos <= std::min(n1, n2); ++pos) {
				auto c = b;
				if (pos < n2) { c[pos] = TypeParam(c[pos] + 1u); } // c[pos] > a[pos] unless it wrapped

				const std::list<TypeParam> la(a.begin(), a.end()), lc(c.begin(), c.end());
				const bool e = std::lexicographical_compare(a.begin(), a.end(), c.begin(), c.end());
				const auto e3 = std::lexicographical_compare_three_way(a.begin(), a.end(), c.begin(), c.end());
				EXPECT_EQ(cryptanalysislib::lexicographical_compare(a.begin(), a.end(), c.begin(), c.end()), e)
				        << n1 << " " << n2 << " " << pos;
				EXPECT_EQ(cryptanalysislib::lexicographical_compare(c.begin(), c.end(), a.begin(), a.end()),
				          std::lexicographical_compare(c.begin(), c.end(), a.begin(), a.end()));
				EXPECT_EQ(cryptanalysislib::lexicographical_compare(la.begin(), la.end(), lc.begin(), lc.end()), e);
				EXPECT_EQ(cryptanalysislib::lexicographical_compare_three_way(a.begin(), a.end(), c.begin(), c.end()), e3)
				        << n1 << " " << n2 << " " << pos;
				EXPECT_EQ(cryptanalysislib::lexicographical_compare_three_way(la.begin(), la.end(), lc.begin(), lc.end()), e3);
			}
		}
	}
}

REGISTER_TYPED_TEST_SUITE_P(LexicographicalCompare, all_positions);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int32_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, LexicographicalCompare, MyTypes);

/// comparators, other element types
TEST(lexicographical_compare, comparators) {
	const std::vector<uint8_t> a = {1, 2, 200}, b = {1, 2, 100};
	EXPECT_FALSE(cryptanalysislib::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()));
	EXPECT_TRUE(cryptanalysislib::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), std::greater<>()));
	EXPECT_EQ(cryptanalysislib::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end()),
	          std::strong_ordering::greater);
	const auto rev = [](const uint8_t x, const uint8_t y) { return y <=> x; };
	EXPECT_EQ(cryptanalysislib::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end(), rev),
	          std::strong_ordering::less);

	const std::string s = "abc", t = "abd";
	EXPECT_TRUE(cryptanalysislib::lexicographical_compare(s.begin(), s.end(), t.begin(), t.end()));
	const std::vector<double> d1 = {1.0, 2.0}, d2 = {1.0, std::numeric_limits<double>::quiet_NaN()};
	EXPECT_EQ(cryptanalysislib::lexicographical_compare_three_way(d1.begin(), d1.end(), d2.begin(), d2.end()),
	          std::partial_ordering::unordered);
}

/// parallel `lexicographical_compare` against `std::lexicographical_compare`:
/// the first difference in different chunks or none, different lengths,
/// SIMD (uint8_t, uint64_t) and scalar (int32_t, comparator) chunks
template<typename T, typename Compare>
static void check_par(const size_t n, Compare comp) {
	std::vector<T> a(n);
	for (auto &x : a) { x = T(rng()); }
	for (const size_t pos : {size_t(0), n / 7, n / 2, n - 1, n}) {
		for (const size_t len : {n, n - 1, n + 1}) {
			std::vector<T> b(a.begin(), a.begin() + std::min(n, len));
			b.resize(len, T(3));
			if (pos < std::min(n, len)) { b[pos] = T(b[pos] + 1); }
			EXPECT_EQ(cryptanalysislib::lexicographical_compare(par_if(true), a.begin(), a.end(), b.begin(), b.end(), comp),
			          std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), comp)) << n << " " << pos << " " << len;
			EXPECT_EQ(cryptanalysislib::lexicographical_compare(par_if(true), b.begin(), b.end(), a.begin(), a.end(), comp),
			          std::lexicographical_compare(b.begin(), b.end(), a.begin(), a.end(), comp)) << n << " " << pos << " " << len;
		}
	}
}

TEST(lexicographical_compare, multithreaded) {
	const size_t n = (1u << 22u) + 9u;
	check_par<uint8_t>(n, std::less<>());
	check_par<uint64_t>(n, std::less<>());
	check_par<int32_t>(n, std::less<>());
	check_par<int32_t>(n, std::greater<>());
	check_par<uint32_t>(1000, std::less<>()); // sequential fallback

	const std::vector<uint32_t> x = {1, 2, 3}, y = {1, 2, 4};
	EXPECT_TRUE(cryptanalysislib::lexicographical_compare(par_if(true), x.begin(), x.end(), y.begin(), y.end()));
}

/// usable in constant expressions (the SIMD path is not taken there)
TEST(lexicographical_compare, constexpr_compare) {
	constexpr auto r = []() {
		const uint32_t a[3] = {1, 2, 3}, b[4] = {1, 2, 3, 0};
		const bool lt = cryptanalysislib::lexicographical_compare(a, a + 3, b, b + 4);
		const auto c = cryptanalysislib::lexicographical_compare_three_way(b, b + 4, a, a + 3);
		return int(lt) * 10 + int(c > 0);
	}();
	static_assert(r == 11);
	EXPECT_EQ(r, 11);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
