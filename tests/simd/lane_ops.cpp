#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

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
