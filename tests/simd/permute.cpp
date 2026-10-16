#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// a random permutation of [0, N)
template<typename T, const uint32_t N>
static void random_permutation(T *p) {
	for (uint32_t i = 0; i < N; ++i) { p[i] = T(i); }
	for (uint32_t i = N - 1; i > 0; --i) {
		const uint32_t j = rng() % (i + 1u);
		const T t = p[i]; p[i] = p[j]; p[j] = t;
	}
}

/// gather permutation: ret[i] = in[perm[i]]
template<typename S>
static void check_permute_gather() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], p[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		random_permutation<T, N>(p);
		S::store(o, S::permute(S::load(a), S::load(p)));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], a[p[i]]); }
	}
}

/// scatter permutation: ret[perm[i]] = in[i]
template<typename S>
static void check_permute_scatter() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], p[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		random_permutation<T, N>(p);
		S::store(o, S::permute(S::load(a), S::load(p)));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[p[i]], a[i]); }
	}
}

/// 64 bit lanes: 2 bit index per lane, within each group of 4 lanes
template<typename S, const uint32_t imm>
static void check_permute_imm(const S va, const uint64_t *a) {
	alignas(64) uint64_t o[S::LIMBS];
	S::store(o, S::template permute<imm>(va));
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		EXPECT_EQ(o[i], a[(i & ~3u) + ((imm >> (2u * (i % 4u))) & 3u)]);
	}
}

template<typename S>
static void check_permute_u64() {
	constexpr uint32_t N = S::LIMBS;
	alignas(64) uint64_t a[N], o[N];
	for (uint32_t i = 0; i < N; ++i) { a[i] = rng(); }
	const S va = S::load(a);
	for (uint32_t imm = 0; imm < 256; ++imm) {
		S::store(o, S::permute(va, imm));
		for (uint32_t i = 0; i < N; ++i) {
			EXPECT_EQ(o[i], a[(i & ~3u) + ((imm >> (2u * (i % 4u))) & 3u)]);
		}
	}

	check_permute_imm<S, 0b00011011>(va, a);
	check_permute_imm<S, 0b11100100>(va, a);
	check_permute_imm<S, 0b01001110>(va, a);
	check_permute_imm<S, 0b00000000>(va, a);
	check_permute_imm<S, 0b10110001>(va, a);
}

#ifdef USE_ARM
// NOTE: the AVX2 version is an in-lane `vpshufb`
TEST(Permute, neon_uint8x32) {
	check_permute_gather<uint8x32_t>();
}
#endif

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(Permute, simd256) {
	check_permute_scatter<uint16x16_t>();
	check_permute_gather<uint32x8_t>();
	check_permute_u64<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(Permute, simd512) {
	check_permute_gather<uint8x64_t>();
	check_permute_scatter<uint16x32_t>();
	check_permute_gather<uint32x16_t>();
	check_permute_u64<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
