#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "algorithm/mismatch.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class Mismatch : public testing::Test {};
TYPED_TEST_SUITE_P(Mismatch);

/// SIMD kernel: every size 0..200 and every mismatch position (or none)
TYPED_TEST_P(Mismatch, simd_all_positions) {
	for (size_t n = 0; n <= 200; ++n) {
		std::vector<TypeParam> a(n), b;
		for (auto &x : a) { x = TypeParam(rng()); }
		b = a;
		EXPECT_EQ(internal::mismatch_simd_uXX<TypeParam>(a.data(), b.data(), n), n);
		for (size_t pos = 0; pos < n; ++pos) {
			std::vector<TypeParam> c = b;
			c[pos] = TypeParam(c[pos] ^ 1u);
			if (pos + 3 < n) { c[pos + 3] = TypeParam(c[pos + 3] ^ 1u); } // first one wins
			EXPECT_EQ(internal::mismatch_simd_uXX<TypeParam>(a.data(), c.data(), n), pos) << n;
		}
	}
}

/// public sequential and parallel versions against std::mismatch
TYPED_TEST_P(Mismatch, sequential_parallel) {
	constexpr size_t n = 1u << 21u; // above `min_size_per_thread`
	std::vector<TypeParam> a(n);
	for (auto &x : a) { x = TypeParam(rng()); }
	for (const size_t pos : {size_t(0), n / 3, n - 1, n}) {
		std::vector<TypeParam> b = a;
		if (pos < n) { b[pos] = TypeParam(b[pos] ^ 1u); }
		const auto e = std::mismatch(a.begin(), a.end(), b.begin());
		EXPECT_EQ(cryptanalysislib::mismatch(a.begin(), a.end(), b.begin()), e) << pos;
		EXPECT_EQ(cryptanalysislib::mismatch(par_if(true), a.begin(), a.end(), b.begin()), e) << pos;
		EXPECT_EQ(cryptanalysislib::mismatch(par_if(true), a.begin(), a.end(), b.begin(), std::equal_to<TypeParam>()), e) << pos;
	}
}

REGISTER_TYPED_TEST_SUITE_P(Mismatch, simd_all_positions, sequential_parallel);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Mismatch, MyTypes);

/// all `mismatch` overloads (predicate, two ranges, parallel) against std::mismatch
TEST(Mismatch, overloads) {
	const auto p = [](const uint32_t a, const uint32_t b) { return (a & 0xffu) == (b & 0xffu); };
	for (const size_t s : {size_t(0), size_t(1), size_t(100), size_t(1u << 21)}) {
		std::vector<uint32_t> a(s), b(s);
		for (size_t i = 0; i < s; ++i) { a[i] = (uint32_t)i; b[i] = (uint32_t)i; }
		for (const size_t pos : {size_t(0), s / 3, s ? s - 1 : 0, s}) {
			std::vector<uint32_t> c = b;
			if (pos < s) { c[pos] += 1u; }
			// shorter second range
			const size_t s2 = s / 2;
			EXPECT_EQ(cryptanalysislib::mismatch(a.begin(), a.end(), c.begin(), p), std::mismatch(a.begin(), a.end(), c.begin(), p)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::mismatch(a.begin(), a.end(), c.begin(), c.end()), std::mismatch(a.begin(), a.end(), c.begin(), c.end())) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::mismatch(a.begin(), a.end(), c.begin(), c.begin() + s2), std::mismatch(a.begin(), a.end(), c.begin(), c.begin() + s2)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::mismatch(a.begin(), a.end(), c.begin(), c.end(), p), std::mismatch(a.begin(), a.end(), c.begin(), c.end(), p)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::mismatch(par_if(true), a.begin(), a.end(), c.begin(), c.end()), std::mismatch(a.begin(), a.end(), c.begin(), c.end())) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::mismatch(par_if(true), a.begin(), a.end(), c.begin(), c.begin() + s2), std::mismatch(a.begin(), a.end(), c.begin(), c.begin() + s2)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::mismatch(par_if(true), a.begin(), a.end(), c.begin(), c.end(), p), std::mismatch(a.begin(), a.end(), c.begin(), c.end(), p)) << s << " " << pos;
			if (s) {
				const size_t k = cryptanalysislib::internal::mismatch_parallel(par_if(true), a.begin(), s, c.begin(), std::equal_to<>(), 4);
				EXPECT_EQ(k, pos < s ? pos : s) << s << " " << pos;
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
