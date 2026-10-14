#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/substr_search.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib::algorithm::internal;

/// the brute force reference
static size_t ref(const uint8_t *h, const size_t hl, const uint8_t *n, const size_t nl) {
	if (hl < nl) { return not_found_k; }
	for (size_t i = 0; i + nl <= hl; ++i) {
		bool ok = true;
		for (size_t j = 0; j < nl; ++j) { ok &= h[i + j] == n[j]; }
		if (ok) { return i; }
	}
	return not_found_k;
}

/// all versions against the brute force reference. The haystack is
/// allocated with its exact size, so ASan detects reads behind it.
TEST(substr_search, all_versions) {
	for (uint32_t it = 0; it < 20000; ++it) {
		const size_t hl = cryptanalysislib::rng() % 300, nl = 1 + cryptanalysislib::rng() % 40;
		const uint32_t alpha = 1 + cryptanalysislib::rng() % 3;
		uint8_t *h = new uint8_t[hl + (hl == 0)];
		uint8_t *nd = new uint8_t[nl];
		for (size_t i = 0; i < hl; ++i) { h[i] = cryptanalysislib::rng() % alpha; }
		for (size_t i = 0; i < nl; ++i) { nd[i] = cryptanalysislib::rng() % alpha; }
		if (hl >= nl && (cryptanalysislib::rng() & 1u)) {
			const size_t p = cryptanalysislib::rng() % (hl - nl + 1);
			for (size_t i = 0; i < nl; ++i) { h[p + i] = nd[i]; }
		}

		span_t H{h, hl}, N{nd, nl};
		const size_t e = ref(h, hl, nd, nl);
		ASSERT_EQ(naive_substr(H, N), e) << hl << " " << nl;
		ASSERT_EQ(prefix_substr(H, N), e) << hl << " " << nl;
		ASSERT_EQ(swar64_strstr_anysize(H, N), e) << hl << " " << nl;
		ASSERT_EQ(cryptanalysislib::algorithm::substr_search(h, hl, nd, nl), e) << hl << " " << nl;
#if defined(__SSE2__)
		ASSERT_EQ(sse2_strstr_anysize(H, N), e) << hl << " " << nl;
#endif
#ifdef USE_ARM
		ASSERT_EQ(neon_speculative_substr(H, N), e) << hl << " " << nl;
#endif
#ifdef USE_AVX2
		ASSERT_EQ(avx2_prefix_substr(H, N), e) << hl << " " << nl;
		ASSERT_EQ(avx2_speculative_substr(H, N), e) << hl << " " << nl;
		ASSERT_EQ(avx2_hybrid_substr(H, N), e) << hl << " " << nl;
		ASSERT_EQ(avx2_strstr_anysize(H, N), e) << hl << " " << nl;
#endif
#ifdef USE_AVX512F
		ASSERT_EQ(avx512_speculative_substr(H, N), e) << hl << " " << nl;
#endif
#if defined(USE_AVX512F) && defined(USE_AVX512BW)
		ASSERT_EQ(avx512bw_strstr_anysize(H, N), e) << hl << " " << nl;
#endif
		delete[] h;
		delete[] nd;
	}
}

/// a long haystack with the needle at the end, empty needle
TEST(substr_search, long_haystack) {
	std::vector<uint8_t> h(100000, 'a');
	const uint8_t n[] = {'a', 'b', 'a'};
	EXPECT_EQ(cryptanalysislib::algorithm::substr_search(h.data(), h.size(), n, 3), not_found_k);
	h[99997] = 'a'; h[99998] = 'b'; h[99999] = 'a';
	EXPECT_EQ(cryptanalysislib::algorithm::substr_search(h.data(), h.size(), n, 3), 99997u);
	EXPECT_EQ(cryptanalysislib::algorithm::substr_search(h.data(), h.size(), n, 0), 0u);
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
