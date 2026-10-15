#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include "algorithm/zip.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(zip, uint8_t_) {
    using TypeParam = uint8_t;
    using TypeParam2 = uint16_t;
    // all lengths around the SIMD widths (16 on NEON, 32 on AVX2)
    for (size_t s = 0; s <= 100; s++) {
        TypeParam d1[100], d2[100];
        TypeParam2 out[100];
        for (uint32_t i = 0; i < s; i++) {
            // NOTE: different values, so swapped inputs are detected
            d1[i] = i; d2[i] = 255 - 3*i;
        }

        zip_u8(out, d1, d2, s);

        for (size_t i = 0; i < s; i++) {
            const TypeParam2 t = d1[i] | (((TypeParam2)d2[i]) << (sizeof(TypeParam)*8));
            EXPECT_EQ(out[i], t);
        }
    }
}

/// checks the zip function `f(out, in1, in2, n)` against the reference for
/// all sizes around the register widths (AVX2: 4..32, AVX-512: 8..64
/// elements per register), also that nothing is written past the 2n outputs
template<typename T, typename F>
static void check_zip(F f) {
	for (size_t s = 0; s <= 200; ++s) {
		std::vector<T> a(s), b(s), out(2 * s + 1, T(7));
		for (size_t i = 0; i < s; ++i) { a[i] = T(i * 3 + 1); b[i] = T(~(i * 5)); }
		f(out.data(), a.data(), b.data(), s);
		for (size_t i = 0; i < s; ++i) {
			ASSERT_EQ(out[2 * i], a[i]) << s << " " << i;
			ASSERT_EQ(out[2 * i + 1], b[i]) << s << " " << i;
		}
		ASSERT_EQ(out[2 * s], T(7)) << "wrote past the end " << s;
	}
}

#ifdef USE_AVX2
TEST(zip, avx2) {
	check_zip<uint8_t>(cryptanalysislib::internal::zip_u8_avx2);
	check_zip<uint16_t>(cryptanalysislib::internal::zip_u16_avx2);
	check_zip<uint32_t>(cryptanalysislib::internal::zip_u32_avx2);
	check_zip<uint64_t>(cryptanalysislib::internal::zip_u64_avx2);
}
#endif

#ifdef USE_AVX512F
TEST(zip, avx512) {
#ifdef USE_AVX512BW
	check_zip<uint8_t>(cryptanalysislib::internal::zip_u8_avx512);
	check_zip<uint16_t>(cryptanalysislib::internal::zip_u16_avx512);
#endif
	check_zip<uint32_t>(cryptanalysislib::internal::zip_u32_avx512);
	check_zip<uint64_t>(cryptanalysislib::internal::zip_u64_avx512);
}
#endif

/// iterator version: the dispatch to the SIMD functions (all unsigned
/// types), a non contiguous/non integral range and constexpr
template<typename T>
static void zip_iterator(T *out, const T *a, const T *b, const size_t n) {
	const auto r = cryptanalysislib::zip(a, a + n, b, out);
	EXPECT_EQ(r - out, ptrdiff_t(2 * n));
}
TEST(zip, iterator_version) {
	check_zip<uint8_t>(zip_iterator<uint8_t>);
	check_zip<uint16_t>(zip_iterator<uint16_t>);
	check_zip<uint32_t>(zip_iterator<uint32_t>);
	check_zip<uint64_t>(zip_iterator<uint64_t>);

	std::list<std::string> a = {"a", "b"}, b = {"x", "y"};
	std::vector<std::string> out(4);
	const auto r = cryptanalysislib::zip(a.begin(), a.end(), b.begin(), out.begin());
	EXPECT_EQ(r, out.end());
	EXPECT_EQ(out, (std::vector<std::string>{"a", "x", "b", "y"}));

	constexpr auto c = []() { int x[2] = {1, 2}, y[2] = {3, 4}, o[4] = {}; cryptanalysislib::zip(x, x + 2, y, o); return o[0] * 1000 + o[1] * 100 + o[2] * 10 + o[3]; }();
	static_assert(c == 1324);
	constexpr auto d = []() { uint8_t x[2] = {1, 2}, y[2] = {3, 4}, o[4] = {}; cryptanalysislib::zip(x, x + 2, y, o); return o[0] * 1000 + o[1] * 100 + o[2] * 10 + o[3]; }();
	static_assert(d == 1324);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
