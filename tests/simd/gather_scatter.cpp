#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `gather<scale>`/`scatter<scale>` with byte offsets `offset[i] * scale`
template<typename S, const uint32_t scale>
static void check_gather_scatter_scale() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	constexpr uint32_t bytes = N * sizeof(T) * 2u;
	alignas(64) uint8_t buf[bytes + 8];
	alignas(64) T off[N], o[N], a[N];
	for (uint32_t r = 0; r < 200; ++r) {
		for (uint32_t i = 0; i < bytes + 8; ++i) { buf[i] = uint8_t(rng()); }

		// gather: any (also unaligned) offset
		for (uint32_t i = 0; i < N; ++i) { off[i] = T(rng() % ((bytes - sizeof(T)) / scale)); }
		S::store(o, S::template gather<scale>(buf, S::load(off)));
		for (uint32_t i = 0; i < N; ++i) {
			T e;
			memcpy(&e, buf + size_t(off[i]) * scale, sizeof(T));
			EXPECT_EQ(o[i], e);
		}

		// scatter: distinct, non overlapping slots in random order
		for (uint32_t i = 0; i < N; ++i) { off[i] = T(i * sizeof(T) * 2u / scale); a[i] = T(rng()); }
		for (uint32_t i = N - 1; i > 0; --i) {
			const uint32_t j = rng() % (i + 1u);
			const T t = off[i]; off[i] = off[j]; off[j] = t;
		}
		S::template scatter<scale>(buf, S::load(off), S::load(a));
		for (uint32_t i = 0; i < N; ++i) {
			T e;
			memcpy(&e, buf + size_t(off[i]) * scale, sizeof(T));
			EXPECT_EQ(e, a[i]);
		}
	}
}

template<typename S>
static void check_gather_scatter() {
	check_gather_scatter_scale<S, 1>();
	check_gather_scatter_scale<S, sizeof(typename S::limb_type)>();
}

/// 64 bit lanes, 32 bit offsets in `I`
template<typename S, typename I>
static void check_gather_u64_u32_offsets() {
	constexpr uint32_t N = S::LIMBS;
	alignas(64) uint8_t buf[N * 16 + 8];
	alignas(64) uint32_t off[I::LIMBS] = {0};
	alignas(64) uint64_t o[N];
	for (uint32_t r = 0; r < 200; ++r) {
		for (uint32_t i = 0; i < sizeof(buf); ++i) { buf[i] = uint8_t(rng()); }
		for (uint32_t i = 0; i < N; ++i) { off[i] = rng() % (N * 16); }
		I idx;
		memcpy(&idx, off, sizeof(off));
		S::store(o, S::gather(buf, idx));
		for (uint32_t i = 0; i < N; ++i) {
			uint64_t e;
			memcpy(&e, buf + off[i], 8);
			EXPECT_EQ(o[i], e);
		}
	}
}

/// 64 bit lanes, 8 bit offsets in `I`
template<typename S, typename I>
static void check_gather_u64_u8_offsets() {
	constexpr uint32_t N = S::LIMBS;
	alignas(64) uint8_t buf[256 + 8];
	alignas(64) uint8_t off[I::LIMBS] = {0};
	alignas(64) uint64_t o[N];
	for (uint32_t r = 0; r < 200; ++r) {
		for (uint32_t i = 0; i < sizeof(buf); ++i) { buf[i] = uint8_t(rng()); }
		for (uint32_t i = 0; i < N; ++i) { off[i] = uint8_t(rng()); }
		S::store(o, S::gather(buf, I::load(off)));
		for (uint32_t i = 0; i < N; ++i) {
			uint64_t e;
			memcpy(&e, buf + off[i], 8);
			EXPECT_EQ(o[i], e);
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(GatherScatter, simd128) {
	check_gather_scatter<_uint8x16_t>();
	check_gather_scatter<_uint16x8_t>();
	check_gather_scatter<_uint32x4_t>();
	check_gather_scatter<_uint64x2_t>();
}

TEST(GatherScatter, simd256) {
	check_gather_scatter<uint8x32_t>();
	check_gather_scatter<uint16x16_t>();
	check_gather_scatter<uint32x8_t>();
	check_gather_scatter<uint64x4_t>();
	check_gather_u64_u32_offsets<uint64x4_t, _uint32x4_t>();
	check_gather_u64_u8_offsets<uint64x4_t, uint8x32_t>();
}
#endif

#ifdef USE_AVX512F
TEST(GatherScatter, simd512) {
	check_gather_scatter<uint8x64_t>();
	check_gather_scatter<uint16x32_t>();
	check_gather_scatter<uint32x16_t>();
	check_gather_scatter<uint64x8_t>();
	check_gather_u64_u32_offsets<uint64x8_t, uint32x8_t>();
	check_gather_u64_u8_offsets<uint64x8_t, uint8x64_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
