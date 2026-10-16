#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// the functions of the SIMD backends, which `TxN_t` implements itself,
/// against a scalar reference. `S` is the generic `TxN_t` (no
/// specialisation which derives from a backend type).
template<typename S>
static void check_txn() {
	using T = typename S::limb_type;
	using U = std::make_unsigned_t<T>;
	using I = std::make_signed_t<T>;
	using M = typename S::mask_type;
	constexpr uint32_t N = S::LIMBS, B = sizeof(T) * 8u;
	alignas(64) T a[N], b[N], o[N], p[N + 1];

	for (uint32_t r = 0; r < 500; ++r) {
		for (uint32_t i = 0; i < N; ++i) {
			a[i] = T(rng() >> (rng() % 64u));
			b[i] = (i % 3u == 0) ? T(0) : T(rng());
		}
		if (r == 0) { a[0] = 0; a[N - 1] = T(-1); }
		M m = 0;
		for (uint32_t i = 0; i < N; ++i) { m |= M(rng() & 1u) << i; }
		const S va = S::load(a), vb = S::load(b);

		// set_bit
		S t = S::set1(T(-1));
		t.set_bit(r % B);
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(U(t.d[i]), U(U(1) << (r % B))); }

		// maskz loads
		S::store(o, S::maskz_unaligned_load(a, m));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], ((m >> i) & 1u) ? a[i] : T(0)); }
		S::store(o, S::maskz_aligned_load(a, m));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], ((m >> i) & 1u) ? a[i] : T(0)); }

		// div
		const T d = T(1u + (rng() % 100u));
		S::store(o, S::div(va, d));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(a[i] / d)); }

		// lzcnt/tzcnt/clz/ctz
		S::store(o, S::lzcnt(va));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(a[i] ? __builtin_clzll(uint64_t(U(a[i]))) - (64u - B) : B)); }
		S::store(o, S::clz(va));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(a[i] ? __builtin_clzll(uint64_t(U(a[i]))) - (64u - B) : B)); }
		S::store(o, S::tzcnt(va));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(a[i] ? __builtin_ctzll(uint64_t(U(a[i]))) : B)); }
		S::store(o, S::ctz(va));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(a[i] ? __builtin_ctzll(uint64_t(U(a[i]))) : B)); }

		// conflict
		if constexpr (N <= B) {
			alignas(64) T c[N];
			for (uint32_t i = 0; i < N; ++i) { c[i] = T(rng() % 4u); }
			S::store(o, S::conflict(S::load(c)));
			for (uint32_t i = 0; i < N; ++i) {
				uint64_t e = 0;
				for (uint32_t j = 0; j < i; ++j) { e |= uint64_t(c[j] == c[i]) << j; }
				EXPECT_EQ(uint64_t(U(o[i])), e);
			}
		}

		// compress
		T e[N];
		uint32_t k = 0;
		for (uint32_t i = 0; i < N; ++i) {
			if ((m >> i) & 1u) { e[k++] = a[i]; }
		}
		S::store(o, S::compress(va, m));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], i < k ? e[i] : T(0)); }
		S::store(o, S::compress(va, vb, m));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], i < k ? e[i] : b[i]); }
		for (uint32_t i = 0; i < N + 1; ++i) { p[i] = T(0x5a); }
		S::compress(p, va, m);
		for (uint32_t i = 0; i < N + 1; ++i) { EXPECT_EQ(p[i], i < k ? e[i] : T(0x5a)); }

		// test
		for (uint32_t bp = 0; bp < B; ++bp) {
			M te = 0;
			for (uint32_t i = 0; i < N; ++i) { te |= M((U(a[i]) >> bp) & 1u) << i; }
			EXPECT_TRUE(S::test(va, bp) == te);
		}

		// reduce_min/reduce_max, min/max
		EXPECT_EQ(S::reduce_min(va), *std::min_element(a, a + N));
		EXPECT_EQ(S::reduce_max(va), *std::max_element(a, a + N));
		S::store(o, S::min(va, vb));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], std::min(a[i], b[i])); }
		S::store(o, S::max(va, vb));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], std::max(a[i], b[i])); }

		// hadd_epu*, histogram_epi4
		const auto hadd = [](const T x, const uint32_t bits) {
			uint64_t s = 0;
			for (uint32_t j = 0; j < B; j += bits) { s += (uint64_t(U(x)) >> j) & ((1ull << bits) - 1ull); }
			return T(s);
		};
		if constexpr (sizeof(T) >= 2) {
			S::store(o, S::hadd_epu8(va));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], hadd(a[i], 8)); }
		}
		if constexpr (sizeof(T) >= 4) {
			S::store(o, S::hadd_epu16(va));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], hadd(a[i], 16)); }
		}
		if constexpr (sizeof(T) == 8) {
			S::store(o, S::hadd_epu32(va));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], hadd(a[i], 32)); }
		}
		const uint8_t nib = uint8_t(r % 16u);
		S::store(o, S::histogram_epi4(va, nib));
		for (uint32_t i = 0; i < N; ++i) {
			uint32_t s = 0;
			for (uint32_t j = 0; j < B; j += 4) { s += ((uint64_t(U(a[i])) >> j) & 0xFu) == nib; }
			EXPECT_EQ(o[i], T(s));
		}

		// pack
		S::store(o, S::pack(m));
		k = 0;
		for (uint32_t i = 0; i < N; ++i) {
			if ((m >> i) & 1u) { EXPECT_EQ(o[k++], T(i)); }
		}
		for (; k < N; ++k) { EXPECT_EQ(o[k], T(0)); }

		// cvtepu8
		if constexpr (N <= 16u) {
			alignas(16) uint8_t c[16];
			for (uint32_t i = 0; i < 16; ++i) { c[i] = uint8_t(rng()); }
			S::store(o, S::cvtepu8(_uint8x16_t::load(c)));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], T(c[i])); }
		}

		// transpose
		if constexpr ((sizeof(T) == 1) && ((N % 8u) == 0)) {
			S::store(o, S::transpose(va));
			for (uint32_t w = 0; w < N; w += 8) {
				for (uint32_t j = 0; j < 8; ++j) {
					for (uint32_t i = 0; i < 8; ++i) {
						EXPECT_EQ((U(o[w + j]) >> i) & 1u, (U(a[w + 7 - i]) >> j) & 1u);
					}
				}
			}
		}

		// comp_sign
		S::store(o, S::comp_sign(va, vb));
		for (uint32_t i = 0; i < N; ++i) {
			const I bi = I(b[i]);
			EXPECT_EQ(o[i], bi == 0 ? T(0) : (bi < 0 ? T(U(0) - U(a[i])) : a[i]));
		}

		// permute(in, imm)
		if constexpr ((N % 4u) == 0) {
			const uint32_t imm = r % 256u;
			S::store(o, S::permute(va, imm));
			for (uint32_t i = 0; i < N; ++i) {
				EXPECT_EQ(o[i], a[(i & ~3u) + ((imm >> (2u * (i % 4u))) & 3u)]);
			}
			S::store(o, S::template permute<0b00011011>(va));
			for (uint32_t i = 0; i < N; ++i) {
				EXPECT_EQ(o[i], a[(i & ~3u) + ((0b00011011u >> (2u * (i % 4u))) & 3u)]);
			}
		}
	}
}

TEST(TxN, uint8) {
	check_txn<TxN_t<uint8_t, 8>>();
	check_txn<TxN_t<uint8_t, 40>>();
	check_txn<TxN_t<uint8_t, 96>>();
}

TEST(TxN, uint16) {
	check_txn<TxN_t<uint16_t, 12>>();
}

TEST(TxN, int16) {
	check_txn<TxN_t<int16_t, 40>>();
}

TEST(TxN, uint32) {
	check_txn<TxN_t<uint32_t, 20>>();
	check_txn<TxN_t<uint32_t, 32>>();
}

TEST(TxN, uint64) {
	check_txn<TxN_t<uint64_t, 12>>();
}

TEST(TxN, int64) {
	check_txn<TxN_t<int64_t, 12>>();
}

TEST(TxN, int32) {
	check_txn<TxN_t<int32_t, 9>>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
