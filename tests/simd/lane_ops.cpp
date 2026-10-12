#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include "algorithm/countl.h"
#include "algorithm/countr.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// compares the lane operations of the unsigned SIMD type `S` against a
/// scalar reference on random inputs
template<typename S>
static void check_lane_ops() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS, B = sizeof(T) * 8u;
	alignas(64) T a[N], b[N], o[N], pm[N];
	for (uint32_t r = 0; r < 500; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			a[i] = T(rng() >> (rng() % 64u));
			b[i] = T(rng());
			pm[i] = T(rng() % N);
		}

		const S va = S::load(a);
		if constexpr (requires { S::reduce_min(va); }) {
			EXPECT_EQ(S::reduce_min(va), *std::min_element(a, a + N));
			EXPECT_EQ(S::reduce_max(va), *std::max_element(a, a + N));
		}

		if constexpr (requires { S::lzcnt(va); }) {
			S::store(o, S::lzcnt(va));
			for (uint32_t i = 0; i < N; ++i) {
				const T e = a[i] ? T(__builtin_clzll(uint64_t(a[i])) - (64u - B)) : T(B);
				EXPECT_EQ(o[i], e);
			}
		}

		if constexpr (requires { S::tzcnt(va); }) {
			S::store(o, S::tzcnt(va));
			for (uint32_t i = 0; i < N; ++i) {
				const T e = a[i] ? T(__builtin_ctzll(uint64_t(a[i]))) : T(B);
				EXPECT_EQ(o[i], e);
			}
		}

		const T d = T(1u + (rng() % 200u));
		S::store(o, S::div(va, d));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(a[i] / d)); }

		if constexpr (requires { S::test(va, 0u); }) {
			const uint32_t bp = rng() % B;
			uint64_t m = 0;
			for (uint32_t i = 0; i < N; ++i) { m |= uint64_t((a[i] >> bp) & 1u) << i; }
			EXPECT_EQ(uint64_t(S::test(va, bp)), m);
		}

		// gather: ret[i] = in[perm[i]] (the 16 bit types scatter, see `permute`)
		if constexpr (requires { S::permute(va, va); } && (sizeof(T) != 2)) {
			S::store(o, S::permute(va, S::load(pm)));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], a[pm[i]]); }
		}
		(void) b;
	}
}

#ifdef USE_AVX2
TEST(LaneOps, avx2_div) {
	check_lane_ops<uint16x16_t>();
	check_lane_ops<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(LaneOps, avx512_uint8x64) { check_lane_ops<uint8x64_t>(); }
TEST(LaneOps, avx512_uint16x32) { check_lane_ops<uint16x32_t>(); }
TEST(LaneOps, avx512_uint32x16) { check_lane_ops<uint32x16_t>(); }
TEST(LaneOps, avx512_uint64x8) { check_lane_ops<uint64x8_t>(); }
#endif

/// division by an invariant scalar: all divisors for 8 bit lanes, many
/// (incl. 1, powers of two, max) for 16 and 32 bit lanes, with extreme numerators
template<typename S>
static void check_div() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	constexpr T MAX = T(-1);
	alignas(64) T a[N], o[N];

	std::vector<T> divisors;
	if constexpr (sizeof(T) == 1) {
		for (uint32_t d = 1; d <= MAX; ++d) { divisors.push_back(T(d)); }
	} else {
		for (uint32_t i = 0; i < sizeof(T) * 8u; ++i) {
			const T p = T(T(1) << i);
			divisors.push_back(p);
			if (p > 1) { divisors.push_back(T(p - 1u)); }
			divisors.push_back(T(p + 1u));
		}
		divisors.push_back(MAX);
		for (uint32_t i = 0; i < 300; ++i) { divisors.push_back(T(1u + (rng() % MAX))); }
	}

	for (const T d : divisors) {
		if (d == 0) { continue; }
		for (uint32_t r = 0; r < 4; ++r) {
			for (uint32_t i = 0; i < N; ++i) {
				a[i] = (i == 0) ? MAX : (i == 1) ? T(0) : (i == 2) ? T(d - 1u) : (i == 3) ? d : T(rng());
			}
			S::store(o, S::div(S::load(a), d));
			for (uint32_t i = 0; i < N; ++i) { ASSERT_EQ(o[i], T(a[i] / d)) << uint64_t(a[i]) << "/" << uint64_t(d); }
		}
	}
}

#ifdef USE_AVX2
TEST(Div, avx2) {
	check_div<uint8x32_t>();
	check_div<uint16x16_t>();
	check_div<uint32x8_t>();
	check_div<uint64x4_t>();
	check_div<cryptanalysislib::_uint8x16_t>();
	check_div<cryptanalysislib::_uint16x8_t>();
	check_div<cryptanalysislib::_uint32x4_t>();
	check_div<int16x16_t>();
}
#endif

#ifdef USE_AVX512F
TEST(Div, avx512) {
	check_div<uint8x64_t>();
	check_div<uint16x32_t>();
	check_div<uint32x16_t>();
	check_div<uint64x8_t>();
}
#endif

#ifdef USE_AVX2
/// clz/ctz of all 16 bit values (and 0)
TEST(LaneOps, avx2_uint16x16_clz_ctz_exhaustive) {
	using S = uint16x16_t;
	alignas(32) uint16_t a[16], o1[16], o2[16];
	for (uint32_t base = 0; base < 65536u; base += 16) {
		for (uint32_t i = 0; i < 16; ++i) { a[i] = uint16_t(base + i); }
		const S v = S::load(a);
		S::store(o1, S::clz(v));
		S::store(o2, S::ctz(v));
		for (uint32_t i = 0; i < 16; ++i) {
			const uint16_t x = a[i];
			ASSERT_EQ(o1[i], cryptanalysislib::countl_zero(x)) << x;
			ASSERT_EQ(o2[i], cryptanalysislib::countr_zero(x)) << x;
		}
	}
}
#endif

#ifdef USE_AVX2
TEST(LaneOps, avx2_uint16x16_conflict) {
	using S = uint16x16_t;
	alignas(32) uint16_t a[16], o[16];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < 16; ++i) { a[i] = uint16_t(rng() % 5u); }
		S::store(o, S::conflict(S::load(a)));
		for (uint32_t i = 0; i < 16; ++i) {
			uint16_t e = 0;
			for (uint32_t j = 0; j < i; ++j) { e |= uint16_t((a[j] == a[i]) << j); }
			ASSERT_EQ(o[i], e);
		}
	}
}
#endif

/// masked load: lanes not in the mask are zero (all masks)
template<typename S>
static void check_maskz_load() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], o[N];
	for (uint32_t m = 0; m < (1u << N); ++m) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		S::store(o, S::maskz_unaligned_load(a, Mask<N>(MaskType<N>(m))));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], ((m >> i) & 1u) ? a[i] : T(0)); }
		S::store(o, S::maskz_aligned_load(a, Mask<N>(MaskType<N>(m))));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], ((m >> i) & 1u) ? a[i] : T(0)); }
	}
}

#ifdef USE_AVX2
TEST(LaneOps, avx2_maskz_load) {
	check_maskz_load<uint64x4_t>();
	check_maskz_load<uint32x8_t>();
}
#endif

/// comparisons, min/max and the constexpr load, for signed and unsigned types
template<typename S>
static void check_cmp_ops() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], b[N], o[N];
	const auto mask = [](const bool c) { return T(c ? -1 : 0); };
	for (uint32_t r = 0; r < 200; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			a[i] = T(rng());
			b[i] = (i % 3u == 0) ? a[i] : T(rng());
		}

		const S va = S::load(a), vb = S::load(b);
		const uint64_t gt = S::gt(va, vb), ge = S::ge(va, vb),
		               lt = S::lt(va, vb), le = S::le(va, vb);
		for (uint32_t i = 0; i < N; ++i) {
			EXPECT_EQ((gt >> i) & 1u, a[i] >  b[i]);
			EXPECT_EQ((ge >> i) & 1u, a[i] >= b[i]);
			EXPECT_EQ((lt >> i) & 1u, a[i] <  b[i]);
			EXPECT_EQ((le >> i) & 1u, a[i] <= b[i]);
		}

		S::store(o, S::gt_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] >  b[i])); }
		S::store(o, S::ge_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] >= b[i])); }
		S::store(o, S::lt_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] <  b[i])); }
		S::store(o, S::le_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] <= b[i])); }
		S::store(o, S::eq_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] == b[i])); }
		S::store(o, S::cmp_(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], mask(a[i] != b[i])); }
		S::store(o, S::min(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], std::min(a[i], b[i])); }
		S::store(o, S::max(va, vb)); for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], std::max(a[i], b[i])); }
		EXPECT_EQ(S::reduce_min(va), *std::min_element(a, a + N));
		EXPECT_EQ(S::reduce_max(va), *std::max_element(a, a + N));
	}

	// constexpr load of negative values
	constexpr S c = []() {
		T t[S::LIMBS] = {};
		for (uint32_t i = 0; i < S::LIMBS; ++i) { t[i] = T(-1 - int(i)); }
		return S::load(t);
	}();
	for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(c[i], T(-1 - int(i))); }
}

#ifdef USE_AVX512F
TEST(CmpOps, avx512_unsigned) {
	check_cmp_ops<uint8x64_t>();
	check_cmp_ops<uint16x32_t>();
	check_cmp_ops<uint32x16_t>();
	check_cmp_ops<uint64x8_t>();
}
TEST(CmpOps, avx512_signed) {
	check_cmp_ops<int8x64_t>();
	check_cmp_ops<int16x32_t>();
	check_cmp_ops<int32x16_t>();
	check_cmp_ops<int64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
