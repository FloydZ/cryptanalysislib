#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "sort/heap/is_heap.h"
#include "sort/heap/push_heap.h"

using ::testing::InitGoogleTest;

/// a random heap and a copy with the heap property violated at one place
static std::vector<int32_t> random_heap(const size_t n) {
	std::vector<int32_t> v(n);
	for (auto &x : v) { x = int32_t(cryptanalysislib::rng() % 1000u); }
	std::make_heap(v.begin(), v.end());
	return v;
}

TEST(Heap, is_heap) {
	for (size_t n = 0; n < 300; n += (n < 40) ? 1 : 17) {
		for (uint32_t r = 0; r < 20; ++r) {
			auto v = random_heap(n);
			if ((r & 1u) && (n > 1)) {
				// break it: a child larger than its parent
				const size_t c = 1 + (cryptanalysislib::rng() % (n - 1));
				v[c] = v[(c - 1) / 2] + 1;
			}

			const bool e = std::is_heap(v.begin(), v.end());
			EXPECT_EQ(is_heap_fwd(v.begin(), v.end(), std::less<>{}), e);
			EXPECT_EQ(is_heap_rnd(v.begin(), v.end(), std::less<>{}), e);
#ifdef USE_AVX2
			EXPECT_EQ(is_heap_sse_epi32(v.data(), v.data() + n), e);
			EXPECT_EQ(is_heap_avx2_epi32(v.data(), v.data() + n), e);
#endif
#ifdef USE_AVX512F
			EXPECT_EQ(is_heap_avx512_epi32(v.data(), v.data() + n), e);
#endif
		}
	}
}

TEST(Heap, push_heap) {
	for (const size_t n : {1ul, 2ul, 10ul, 255ul, 256ul, 257ul, 1000ul, 5000ul, 32768ul, 40000ul, 65535ul, 65536ul}) {
		for (uint32_t r = 0; r < 20; ++r) {
			auto v = random_heap(n - 1);
			// also values larger than all others (new root) and negative ones
			const uint32_t k = cryptanalysislib::rng() % 3u;
			v.push_back(k == 0 ? 5000 : (k == 1 ? -5 : int32_t(cryptanalysislib::rng() % 2000u)));
			auto w = v;
			std::push_heap(w.begin(), w.end());

			auto a = v;
			push_heap_scalar(a.data(), a.data() + n, std::less<>{});
			EXPECT_TRUE(std::is_heap(a.begin(), a.end()));
			EXPECT_EQ(a, w);

#ifdef USE_AVX2
			auto b = v;
			push_heap_avx2(b.data(), n);
			EXPECT_TRUE(std::is_heap(b.begin(), b.end()));
			EXPECT_TRUE(std::is_permutation(b.begin(), b.end(), w.begin()));
#endif
#ifdef USE_AVX512F
			auto c = v;
			push_heap_avx512(c.data(), n);
			EXPECT_TRUE(std::is_heap(c.begin(), c.end()));
			EXPECT_TRUE(std::is_permutation(c.begin(), c.end(), w.begin()));
#endif
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	cryptanalysislib::rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
