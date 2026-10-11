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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
