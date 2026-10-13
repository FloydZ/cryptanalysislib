#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "algorithm/set.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class Set : public testing::Test {};
TYPED_TEST_SUITE_P(Set);

/// sorted random multiset with values in [0, mod)
template<typename T>
static std::vector<T> sorted_random(const size_t n, const uint32_t mod) {
	std::vector<T> v(n);
	for (auto &x : v) { x = T(rng() % mod); }
	std::sort(v.begin(), v.end());
	return v;
}

/// the parallel set operations against std:: on multisets: few distinct
/// values (long runs of equal elements across chunk boundaries) and many
#define CHECK_SET_OP(NAME)                                                                   \
	{                                                                                        \
		std::vector<uint32_t> out(a.size() + b.size()), e(a.size() + b.size());              \
		const auto r = cryptanalysislib::NAME(par_if(true), a.begin(), a.end(), b.begin(), b.end(), out.begin()); \
		const auto re = std::NAME(a.begin(), a.end(), b.begin(), b.end(), e.begin());        \
		ASSERT_EQ(r - out.begin(), re - e.begin()) << #NAME << " " << mod;                   \
		ASSERT_TRUE(std::equal(out.begin(), r, e.begin())) << #NAME << " " << mod;           \
	}

TEST(set, parallel_operations) {
	constexpr size_t n = 1u << 20u;
	for (const uint32_t mod : {3u, 100u, 1u << 30}) {
		const auto a = sorted_random<uint32_t>(n, mod), b = sorted_random<uint32_t>(n / 2, mod);
		CHECK_SET_OP(set_difference)
		CHECK_SET_OP(set_intersection)
		CHECK_SET_OP(set_union)
		CHECK_SET_OP(set_symmetric_difference)

		EXPECT_EQ(cryptanalysislib::includes(par_if(true), a.begin(), a.end(), b.begin(), b.end()),
		          std::includes(a.begin(), a.end(), b.begin(), b.end())) << mod;
		std::vector<uint32_t> sub;
		for (size_t i = 0; i < n; i += 3) { sub.push_back(a[i]); }
		EXPECT_TRUE(cryptanalysislib::includes(par_if(true), a.begin(), a.end(), sub.begin(), sub.end())) << mod;
	}
}

/// set_difference_uXX_simd against std::set_difference (multiset semantics)
TYPED_TEST_P(Set, set_difference_simd) {
	for (const uint32_t mod : {4u, 50u, 100000u}) {
		for (size_t n1 = 0; n1 <= 140; n1 += 7) {
			for (size_t n2 = 0; n2 <= 140; n2 += 9) {
				const auto a = sorted_random<TypeParam>(n1, mod), b = sorted_random<TypeParam>(n2, mod);
				std::vector<TypeParam> out(n1), e(n1);
				const size_t r = internal::set_difference_uXX_simd(a.data(), n1, b.data(), n2, out.data());
				const auto re = std::set_difference(a.begin(), a.end(), b.begin(), b.end(), e.begin());
				ASSERT_EQ(r, size_t(re - e.begin())) << n1 << " " << n2 << " " << mod;
				out.resize(r); e.resize(r);
				ASSERT_EQ(out, e) << n1 << " " << n2 << " " << mod;
			}
		}
	}
}

/// includes_uXX_simd against std::includes: random pairs, and subsets of `a`
TYPED_TEST_P(Set, includes_simd) {
	for (const uint32_t mod : {4u, 50u, 100000u}) {
		for (size_t n1 = 0; n1 <= 140; n1 += 7) {
			for (size_t n2 = 0; n2 <= 140; n2 += 9) {
				const auto a = sorted_random<TypeParam>(n1, mod), b = sorted_random<TypeParam>(n2, mod);
				ASSERT_EQ(internal::includes_uXX_simd(a.data(), n1, b.data(), n2),
				          std::includes(a.begin(), a.end(), b.begin(), b.end())) << n1 << " " << n2 << " " << mod;

				// a sub-multiset of `a` (every 3rd element) must be included
				std::vector<TypeParam> sub;
				for (size_t i = 0; i < n1; i += 3) { sub.push_back(a[i]); }
				ASSERT_TRUE(internal::includes_uXX_simd(a.data(), n1, sub.data(), sub.size())) << n1 << " " << mod;
			}
		}
	}
}

REGISTER_TYPED_TEST_SUITE_P(Set, set_difference_simd, includes_simd);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Set, MyTypes);

/// sequential set operations (with and without comparator) against std::
TEST(set, sequential_operations) {
	for (const uint32_t mod : {3u, 50u, 1000u}) {
		std::vector<uint32_t> a(400), b(300);
		for (auto &x : a) { x = rng<uint32_t>() % mod; }
		for (auto &x : b) { x = rng<uint32_t>() % mod; }
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());
		std::vector<uint32_t> sub;
		for (size_t i = 0; i < a.size(); i += 3) { sub.push_back(a[i]); }
		std::vector<uint32_t> o(1000), e(1000);
		const auto gt = std::greater<>();
		std::vector<uint32_t> ra(a.rbegin(), a.rend()), rb(b.rbegin(), b.rend());

		EXPECT_EQ(cryptanalysislib::includes(a.begin(), a.end(), b.begin(), b.end()), std::includes(a.begin(), a.end(), b.begin(), b.end()));
		EXPECT_TRUE(cryptanalysislib::includes(a.begin(), a.end(), sub.begin(), sub.end()));
		EXPECT_EQ(cryptanalysislib::includes(ra.begin(), ra.end(), rb.begin(), rb.end(), gt), std::includes(ra.begin(), ra.end(), rb.begin(), rb.end(), gt));

#define CHECK_SET_OP(OP) {                                                                                   \
			auto r1 = cryptanalysislib::OP(a.begin(), a.end(), b.begin(), b.end(), o.begin());             \
			auto r2 = std::OP(a.begin(), a.end(), b.begin(), b.end(), e.begin());                          \
			ASSERT_EQ(r1 - o.begin(), r2 - e.begin()) << #OP << " " << mod;                                \
			EXPECT_TRUE(std::equal(o.begin(), r1, e.begin())) << #OP << " " << mod;                        \
			r1 = cryptanalysislib::OP(ra.begin(), ra.end(), rb.begin(), rb.end(), o.begin(), gt);          \
			r2 = std::OP(ra.begin(), ra.end(), rb.begin(), rb.end(), e.begin(), gt);                       \
			ASSERT_EQ(r1 - o.begin(), r2 - e.begin()) << #OP << " cmp " << mod;                            \
			EXPECT_TRUE(std::equal(o.begin(), r1, e.begin())) << #OP << " cmp " << mod; }
		CHECK_SET_OP(set_difference)
		CHECK_SET_OP(set_intersection)
		CHECK_SET_OP(set_symmetric_difference)
		CHECK_SET_OP(set_union)
#undef CHECK_SET_OP
	}
}

/// lower/upper positions of the parallel set operations
TEST(set, internal_positions) {
	const std::vector<uint32_t> a = {1, 2, 2, 2, 5, 7, 7, 9};
	for (uint32_t v = 0; v < 11; ++v) {
		EXPECT_EQ(cryptanalysislib::internal::set_lower_pos(a.begin(), 0, a.size(), v), size_t(std::lower_bound(a.begin(), a.end(), v) - a.begin())) << v;
		EXPECT_EQ(cryptanalysislib::internal::set_upper_pos(a.begin(), 0, a.size(), v), size_t(std::upper_bound(a.begin(), a.end(), v) - a.begin())) << v;
		EXPECT_EQ(cryptanalysislib::internal::set_lower_pos(a.begin(), 2, 6, v), size_t(std::lower_bound(a.begin() + 2, a.begin() + 6, v) - a.begin())) << v;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
