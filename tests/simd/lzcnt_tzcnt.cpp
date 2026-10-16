#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `lzcnt`/`tzcnt` against `__builtin_clzll`/`__builtin_ctzll`, for random
/// values, 0, all powers of two and their neighbours
template<typename S>
static void check_lzcnt_tzcnt() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS, B = sizeof(T) * 8u;
	alignas(64) T a[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			const uint32_t k = (r + i) % (3u * B);
			if (k < B) {
				a[i] = T(T(1) << k);
			} else if (k < 2u * B) {
				a[i] = T((T(1) << (k - B)) - 1u);
			} else {
				a[i] = T(rng() >> (rng() % 64u));
			}
		}
		if (r == 0) { a[0] = 0; a[N - 1] = T(-1); }

		const S va = S::load(a);
		S::store(o, S::lzcnt(va));
		for (uint32_t i = 0; i < N; ++i) {
			const T e = a[i] ? T(__builtin_clzll(uint64_t(a[i])) - (64u - B)) : T(B);
			EXPECT_EQ(o[i], e);
		}

		S::store(o, S::tzcnt(va));
		for (uint32_t i = 0; i < N; ++i) {
			const T e = a[i] ? T(__builtin_ctzll(uint64_t(a[i]))) : T(B);
			EXPECT_EQ(o[i], e);
		}
	}
}

/// `clz` (if `with_clz`) and `ctz` against `__builtin_clzll`/`__builtin_ctzll`
template<typename S, const bool with_clz>
static void check_clz_ctz() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS, B = sizeof(T) * 8u;
	alignas(64) T a[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			a[i] = (i == r % N) ? T(0) : T(rng() >> (rng() % 64u));
		}

		const S va = S::load(a);
		if constexpr (with_clz) {
			S::store(o, S::clz(va));
			for (uint32_t i = 0; i < N; ++i) {
				const T e = a[i] ? T(__builtin_clzll(uint64_t(a[i])) - (64u - B)) : T(B);
				EXPECT_EQ(o[i], e);
			}
		}

		S::store(o, S::ctz(va));
		for (uint32_t i = 0; i < N; ++i) {
			const T e = a[i] ? T(__builtin_ctzll(uint64_t(a[i]))) : T(B);
			EXPECT_EQ(o[i], e);
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(LzcntTzcnt, simd256) {
	check_lzcnt_tzcnt<uint8x32_t>();
	check_lzcnt_tzcnt<uint16x16_t>();
	check_lzcnt_tzcnt<uint64x4_t>();
}

TEST(ClzCtz, simd256) {
	check_clz_ctz<uint16x16_t, true>();
	check_clz_ctz<uint32x8_t, false>();
}
#endif

#ifdef USE_AVX512F
TEST(LzcntTzcnt, simd512) {
	check_lzcnt_tzcnt<uint8x64_t>();
	check_lzcnt_tzcnt<uint16x32_t>();
	check_lzcnt_tzcnt<uint64x8_t>();
}

TEST(ClzCtz, simd512) {
	check_clz_ctz<uint16x32_t, true>();
	check_clz_ctz<uint32x16_t, false>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
