#include <algorithm>
#include <bit>
#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <numeric>
#include <vector>

#include "algorithm/rotate.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// every in-place rotation algorithm against std::rotate: all lengths 0..70
/// and every split point `left`
template<typename T, typename F>
static void check_all(F f, const char *name) {
	for (size_t n = 0; n <= 70; ++n) {
		for (size_t left = 0; left <= n; ++left) {
			std::vector<T> v(n), e(n);
			std::iota(v.begin(), v.end(), T(1));
			e = v;
			f(v.data(), left, n - left);
			std::rotate(e.begin(), e.begin() + left, e.end());
			ASSERT_EQ(v, e) << name << " n=" << n << " left=" << left;
		}
	}
}

TEST(rotate, algorithms_uint32) {
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { contrev_rotation(a, l, r); }, "contrev");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { trinity_rotation<uint32_t, 8>(a, l, r); }, "trinity<8>");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { trinity_rotation<uint32_t, 32>(a, l, r); }, "trinity<32>");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { griesmills_rotation(a, l, r); }, "griesmills");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { grail_rotation(a, l, r); }, "grail");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { piston_rotation(a, l, r); }, "piston");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { helix_rotation(a, l, r); }, "helix");
	check_all<uint32_t>([](uint32_t *a, size_t l, size_t r) { drill_rotation(a, l, r); }, "drill");
}

TEST(rotate, algorithms_uint8_uint64) {
	check_all<uint8_t>([](uint8_t *a, size_t l, size_t r) { helix_rotation(a, l, r); }, "helix u8");
	check_all<uint64_t>([](uint64_t *a, size_t l, size_t r) { trinity_rotation<uint64_t, 16>(a, l, r); }, "trinity u64");
}

TEST(rotate, iterator_version) {
	for (size_t n = 0; n <= 40; ++n) {
		for (size_t left = 0; left <= n; ++left) {
			std::vector<int> v(n), e(n);
			std::iota(v.begin(), v.end(), 0);
			e = v;
			const auto r = cryptanalysislib::rotate(v.begin(), v.begin() + left, v.end());
			const auto re = std::rotate(e.begin(), e.begin() + left, e.end());
			ASSERT_EQ(v, e) << n << " " << left;
			ASSERT_EQ(r - v.begin(), re - e.begin()) << n << " " << left;
		}
	}
	std::list<int> l = {1, 2, 3, 4, 5};
	cryptanalysislib::rotate(l.begin(), std::next(l.begin(), 2), l.end());
	EXPECT_EQ(l, (std::list<int>{3, 4, 5, 1, 2}));

	std::vector<int> in = {1, 2, 3, 4}, out(4);
	cryptanalysislib::rotate_copy(in.begin(), in.begin() + 1, in.end(), out.begin());
	EXPECT_EQ(out, (std::vector<int>{2, 3, 4, 1}));
}

/// the block swaps and the stack rotation of `cryptanalysislib::internal`
TEST(rotate, internal_helpers) {
	std::vector<uint32_t> v(20);
	for (size_t i = 0; i < v.size(); ++i) { v[i] = (uint32_t)i; }
	std::vector<uint32_t> a = v, e = v;
	cryptanalysislib::internal::forward_block_swap(a.data(), 2, 10, 5);
	std::swap_ranges(e.begin() + 2, e.begin() + 7, e.begin() + 10);
	EXPECT_EQ(a, e);
	a = v; e = v;
	cryptanalysislib::internal::backward_block_swap(a.data(), 3, 12, 6);
	std::swap_ranges(e.begin() + 3, e.begin() + 9, e.begin() + 12);
	EXPECT_EQ(a, e);

	// rotation of `left + right` elements by `left` (min(left, right) <= MAX_AUX)
	for (size_t left = 0; left <= 8; ++left) {
		for (const size_t right : {size_t(0), size_t(1), size_t(8), size_t(12)}) {
			if (left + right == 0 || left + right > v.size()) { continue; }
			a = v; e = v;
			cryptanalysislib::internal::stack_rotation<uint32_t, 8>(a.data(), left, right);
			std::rotate(e.begin(), e.begin() + left, e.begin() + left + right);
			EXPECT_EQ(a, e) << left << " " << right;
		}
	}
}

/// bit rotations of integers (runtime and consteval versions)
TEST(rotate, bit_rotations) {
	EXPECT_EQ(cryptanalysislib::rotl<uint64_t>(1u, 1), 2u);
	EXPECT_EQ(cryptanalysislib::rotl<uint64_t>(1ull << 63, 1), 1u);
	EXPECT_EQ(cryptanalysislib::rotr<uint64_t>(1u, 1), 1ull << 63);
	EXPECT_EQ(cryptanalysislib::rotl<uint8_t>(0x81u, 1), 0x03u);
	EXPECT_EQ(cryptanalysislib::rotr<uint8_t>(0x81u, 1), 0xc0u);
	EXPECT_EQ(cryptanalysislib::rotl<uint16_t>(0x1234u, 4), 0x2341u);
	EXPECT_EQ(cryptanalysislib::rotr<uint32_t>(0x12345678u, 8), 0x78123456u);
	for (uint32_t k = 0; k < 70; ++k) {
		const uint64_t x = 0x0123456789abcdefull;
		EXPECT_EQ(cryptanalysislib::rotl<uint64_t>(x, k), std::rotl(x, (int)k)) << k;
		EXPECT_EQ(cryptanalysislib::rotr<uint64_t>(x, k), std::rotr(x, (int)k)) << k;
		const uint32_t y = 0x89abcdefu;
		EXPECT_EQ(cryptanalysislib::rotl<uint32_t>(y, k), std::rotl(y, (int)k)) << k;
		EXPECT_EQ(cryptanalysislib::rotr<uint32_t>(y, k), std::rotr(y, (int)k)) << k;
	}
	static_assert(cryptanalysislib::rotl<3>(uint8_t(0x81)) == uint8_t(0x0c));
	static_assert(cryptanalysislib::rotr<3>(uint8_t(0x81)) == uint8_t(0x30));
	static_assert(cryptanalysislib::rotl<8>(uint32_t(0x12345678)) == uint32_t(0x34567812));
	static_assert(cryptanalysislib::rotr<8>(uint32_t(0x12345678)) == uint32_t(0x78123456));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
