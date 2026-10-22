#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "random.h"
#include "container/fq_packed_vector_v2.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// set/get of random numbers against a plain array, and `hash` against the
/// bit string of the numbers
template<const uint32_t n, const uint64_t q, typename T, const bool u>
static void check() {
	using V = FqPackedVectorMeta_v2<n, q, T, u>;
	constexpr uint32_t bits = V::bits_per_number;
	static_assert(V::internal_limbs * V::limb_bits >= n * bits);

	for (uint32_t it = 0; it < 100; ++it) {
		V v;
		std::vector<int64_t> e(n);
		for (uint32_t i = 0; i < n; ++i) {
			const int64_t x = int64_t(rng() % q);
			// also negative numbers for the signed version
			const int64_t y = u ? x : x - int64_t(q / 2u);
			v.set(typename V::DataType(y), i);
			int64_t r = y % int64_t(q);
			if (r < 0) { r += int64_t(q); }
			if (!u && (uint64_t(r) > (q / 2u))) { r -= int64_t(q); }
			e[i] = r;
		}

		for (uint32_t i = 0; i < n; ++i) {
			ASSERT_EQ(int64_t(v.get(i)), e[i]) << i << " " << it;
		}

		// the numbers [l, h) as a bit string
		for (uint32_t l = 0; l < n; ++l) {
			for (uint32_t h = l + 1; (h <= n) && ((h - l) * bits <= 64u); ++h) {
				uint64_t exp = 0;
				for (uint32_t i = l; i < h; ++i) {
					int64_t r = e[i];
					if (r < 0) { r += int64_t(q); }
					exp |= uint64_t(r) << ((i - l) * bits);
				}
				ASSERT_EQ(v.hash(l, h), exp) << l << " " << h;
			}
		}
		if constexpr ((n >= 2) && V::template is_hashable<0, 2>()) {
			const uint64_t h1 = v.template hash<0, 2>();
			ASSERT_EQ(h1, v.hash(0, 2));
		}
	}
}

TEST(FqPackedVector_v2, q3) {
	check<1, 3, uint64_t, true>();
	check<31, 3, uint64_t, true>();
	check<33, 3, uint64_t, true>();
	check<100, 3, uint8_t, true>();
	check<100, 3, uint64_t, false>();
}

TEST(FqPackedVector_v2, q7) {
	// 3 bits per number: the numbers are split over two limbs
	check<22, 7, uint64_t, true>();
	check<50, 7, uint8_t, true>();
	check<50, 7, uint32_t, false>();
}

TEST(FqPackedVector_v2, q251) {
	check<17, 251, uint64_t, true>();
	check<17, 251, uint16_t, false>();
}

TEST(FqPackedVector_v2, constructor) {
	using V = FqPackedVectorMeta_v2<10, 7, uint64_t, true>;
	const V v(V::DataType(12));
	EXPECT_EQ(v.get(0), 5);
	for (uint32_t i = 1; i < 10; ++i) {
		EXPECT_EQ(v.get(i), 0);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
