#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// random lanes: zero or a single bit (msb, lsb or random) set; in a quarter
/// of the runs all lanes are zero, in another quarter none is
template<class S>
static void check_reduce() {
	constexpr uint32_t n = S::LIMBS;
	for (uint32_t it = 0; it < 10000; ++it) {
		S a = S::set1(0);
		const uint32_t mode = rng() % 4;
		for (uint32_t i = 0; i < n; ++i) {
			bool set = rng() & 1u;
			if (mode == 0) { set = false; }
			if (mode == 1) { set = true; }
			// msb, lsb, any bit
			const uint32_t k = rng() % 3;
			const auto bits = sizeof(a[i]) * 8u;
			using L = std::remove_cvref_t<decltype(a[i])>;
			const L v = k == 0 ? L(L(1u) << (bits - 1u)) : (k == 1 ? L(1u) : L(L(1u) << (rng() % bits)));
			a[i] = set ? v : L(0);
		}

		uint32_t c = 0;
		for (uint32_t i = 0; i < n; ++i) { c += a[i] != 0; }

		ASSERT_EQ(all_of(a), c == n);
		ASSERT_EQ(any_of(a), c > 0);
		ASSERT_EQ(none_of(a), c == 0);
		ASSERT_EQ(some_of(a), c > 0 && c < n);

		const auto m = lane_mask(a);
		for (uint32_t i = 0; i < n; ++i) {
			ASSERT_EQ(bool((m >> i) & 1u), a[i] != 0);
		}
		for (uint32_t i = n; i < sizeof(m) * 8u; ++i) {
			ASSERT_EQ(bool((m >> i) & 1u), false);
		}
	}
}

TEST(reduce, comparison) {
	const uint32x8_t a = uint32x8_t::set1(5), b = uint32x8_t::set1(7);
	EXPECT_TRUE(all_of(uint32x8_t::lt_(a, b)));
	EXPECT_TRUE(none_of(uint32x8_t::eq_(a, b)));
	EXPECT_FALSE(some_of(uint32x8_t::eq_(a, a)));

	uint8x32_t c = uint8x32_t::set1(1), d = uint8x32_t::set1(1);
	d[17] = 2;
	EXPECT_TRUE(any_of(uint8x32_t::cmp_(c, d)));
	EXPECT_TRUE(some_of(uint8x32_t::eq_(c, d)));
	EXPECT_FALSE(all_of(uint8x32_t::eq_(c, d)));
}

TEST(reduce, uint8x32_t) { check_reduce<uint8x32_t>(); }
TEST(reduce, uint16x16_t) { check_reduce<uint16x16_t>(); }
TEST(reduce, uint32x8_t) { check_reduce<uint32x8_t>(); }
TEST(reduce, uint64x4_t) { check_reduce<uint64x4_t>(); }
TEST(reduce, int8x32_t) { check_reduce<int8x32_t>(); }
TEST(reduce, int64x4_t) { check_reduce<int64x4_t>(); }
TEST(reduce, _uint8x16_t) { check_reduce<_uint8x16_t>(); }
TEST(reduce, _uint16x8_t) { check_reduce<_uint16x8_t>(); }
TEST(reduce, _uint32x4_t) { check_reduce<_uint32x4_t>(); }
TEST(reduce, _uint64x2_t) { check_reduce<_uint64x2_t>(); }

#ifdef USE_AVX512F
TEST(reduce, uint8x64_t) { check_reduce<uint8x64_t>(); }
TEST(reduce, uint16x32_t) { check_reduce<uint16x32_t>(); }
TEST(reduce, uint32x16_t) { check_reduce<uint32x16_t>(); }
TEST(reduce, uint64x8_t) { check_reduce<uint64x8_t>(); }
#endif

TEST(reduce, TxN_t) {
	check_reduce<TxN_t<uint32_t, 8>>();
	check_reduce<TxN_t<uint64_t, 4>>();
	check_reduce<TxN_t<uint8_t, 3>>();
	check_reduce<TxN_t<uint32_t, 5>>();
	check_reduce<TxN_t<uint16_t, 40>>();
	check_reduce<TxN_t<uint8_t, 64>>();
	check_reduce<TxN_t<uint8_t, 100>>();
	check_reduce<TxN_t<uint64_t, 16>>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
