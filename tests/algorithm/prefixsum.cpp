#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/prefixsum.h"
#include "algorithm/inclusive_scan.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


using namespace cryptanalysislib::algorithm;

#ifdef USE_AVX2
TEST(prefix_sum, int32_simd) {
    constexpr static size_t s = 100;
    using T = int32_t;
    std::vector<T> in; in.resize(s);
    std::ranges::fill(in, 1);
    cryptanalysislib::algorithm::internal::prefixsum_i32_avx2(in.data(), s);
	for (size_t i = 0; i < s; i++) {
		EXPECT_EQ(i + 1, in[i]);
	}
}

TEST(prefix_sum, int32_simd_v2) {
    constexpr static size_t s = 100;
    using T = int32_t;
    std::vector<T> in; in.resize(s);
    std::ranges::fill(in, 1);
    cryptanalysislib::algorithm::internal::prefixsum_i32_avx2_v2(in.data(), s);
	for (size_t i = 0; i < s; i++) {
		EXPECT_EQ(i+1, in[i]);
	}
}
#endif

#ifdef USE_AVX512F

TEST(prefix_sum, u32_simd_avx512) {
    constexpr static size_t s = 100;
    using T = uint32_t;
    std::vector<T> in; in.resize(s);
    std::ranges::fill(in, 1);
    cryptanalysislib::algorithm::internal::prefixsum_u32_avx512(in.data(), s);
	for (size_t i = 0; i < s; i++) {
		EXPECT_EQ(i + 1, in[i]);
	}
}
#endif

/// checks `f` against a scalar prefix sum for all sizes [0, 80]. Each input
/// is allocated with its exact size, so ASan catches out of bounds accesses.
template<typename T, typename F>
static void check_all_sizes(F f) {
	for (size_t n = 0; n <= 80; n++) {
		T *v = new T[n + (n == 0)];
		std::vector<T> e(n);
		for (size_t i = 0; i < n; i++) { e[i] = v[i] = T(i * 7 + 3); }
		for (size_t i = 1; i < n; i++) { e[i] += e[i - 1]; }

		f(v, n);
		for (size_t i = 0; i < n; i++) {
			EXPECT_EQ(e[i], v[i]) << "n=" << n << " i=" << i;
		}
		delete[] v;
	}
}

TEST(prefix_sum, all_sizes) {
#ifdef USE_AVX2
	check_all_sizes<int32_t>([](int32_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_i32_avx2(v, n); });
	check_all_sizes<int32_t>([](int32_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_i32_avx2_v2(v, n); });
	check_all_sizes<uint32_t>([](uint32_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_u32_avx2(v, n); });
#endif
#ifdef USE_AVX512F
	check_all_sizes<uint32_t>([](uint32_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_u32_avx512(v, n); });
#endif
	check_all_sizes<uint32_t>([](uint32_t *v, size_t n) { prefixsum<uint32_t>(v, n); });
	check_all_sizes<uint8_t>([](uint8_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_uXX_simd<uint8_t>(v, n); });
	check_all_sizes<uint16_t>([](uint16_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_uXX_simd<uint16_t>(v, n); });
	check_all_sizes<uint32_t>([](uint32_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_uXX_simd<uint32_t>(v, n); });
	check_all_sizes<uint64_t>([](uint64_t *v, size_t n) { cryptanalysislib::algorithm::internal::prefixsum_uXX_simd<uint64_t>(v, n); });
	check_all_sizes<uint64_t>([](uint64_t *v, size_t n) { prefixsum<uint64_t>(v, n); });
}

TEST(prefix_sum, empty_range) {
	std::vector<uint32_t> v;
	prefixsum(v.begin(), v.end());
	EXPECT_TRUE(v.empty());
}

TEST(avx, prefixsum) {
	constexpr size_t s = 65;
	uint32_t *d1 = (uint32_t *)malloc(s * sizeof(uint32_t));
	uint32_t *d2 = (uint32_t *)malloc(s * sizeof(uint32_t));
	// for (uint32_t i = 0; i < s; i++) { d1[i] = fastrandombytes_uint64() % (1u << 8u); }
	for (uint32_t i = 0; i < s; i++) {
		d1[i] = i;
	}
	memcpy(d2, d1, s * sizeof(uint32_t));

	prefixsum(d1, s);
	for (uint32_t i = 1; i < s; i++) {
		d2[i] += d2[i - 1];
	}

	for (uint32_t i = 0; i < s; i++) {
	EXPECT_EQ(d1[i], d2[i]);
	}
	free(d1); free(d2);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
