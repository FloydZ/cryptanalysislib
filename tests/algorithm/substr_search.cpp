#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/substr_search.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib::algorithm::internal;

/// the brute force reference
static size_t ref(const std::vector<uint8_t> &h, const std::vector<uint8_t> &n) {
	if (h.size() < n.size()) { return not_found_k; }
	for (size_t i = 0; i + n.size() <= h.size(); ++i) {
		bool ok = true;
		for (size_t j = 0; j < n.size(); ++j) { ok &= h[i + j] == n[j]; }
		if (ok) { return i; }
	}
	return not_found_k;
}

TEST(substr_search, all_versions) {
	for (uint32_t it = 0; it < 20000; ++it) {
		const size_t hl = cryptanalysislib::rng() % 200, nl = 1 + cryptanalysislib::rng() % 12;
		const uint32_t alpha = 1 + cryptanalysislib::rng() % 3;
		std::vector<uint8_t> h(hl + 1), nd(nl);
		for (auto &c : h) { c = cryptanalysislib::rng() % alpha; }
		for (auto &c : nd) { c = cryptanalysislib::rng() % alpha; }
		h.resize(hl);
		if (hl >= nl && (cryptanalysislib::rng() & 1u)) {
			const size_t p = cryptanalysislib::rng() % (hl - nl + 1);
			for (size_t i = 0; i < nl; ++i) { h[p + i] = nd[i]; }
		}
		span_t H{h.data(), hl}, N{nd.data(), nl};
		const size_t e = ref(h, nd);
		ASSERT_EQ(naive_substr(H, N), e) << hl << " " << nl;
		ASSERT_EQ(prefix_substr(H, N), e) << hl << " " << nl;
#ifdef USE_ARM
		ASSERT_EQ(neon_speculative_substr(H, N), e) << hl << " " << nl;
#endif
#ifdef USE_AVX2
		ASSERT_EQ(avx2_speculative_substr(H, N), e) << hl << " " << nl;
#endif
	}
}

TEST(substr_search, are_equal) {
	const uint8_t a[] = {1, 2, 3, 4}, b[] = {1, 2, 3, 5};
	EXPECT_TRUE(are_equal(a, b, 0));
	EXPECT_TRUE(are_equal(a, b, 3));
	EXPECT_FALSE(are_equal(a, b, 4));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
