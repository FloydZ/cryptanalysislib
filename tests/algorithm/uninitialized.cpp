#include <gtest/gtest.h>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "algorithm/uninitialized.h"

using ::testing::InitGoogleTest;

/// long strings own heap memory: a missing destructor call is a leak (ASan)
static std::string str(const size_t i) {
	return std::string(40, char('a' + (i % 26)));
}

/// copy/move/fill into uninitialized memory, then destroy
TEST(uninitialized, copy_move_fill_destroy) {
	std::allocator<std::string> alloc;
	for (const size_t n : {size_t(0), size_t(1), size_t(17)}) {
		std::vector<std::string> src(n);
		for (size_t i = 0; i < n; ++i) { src[i] = str(i); }
		const std::list<std::string> lsrc(src.begin(), src.end());

		std::string *p = alloc.allocate(n + 1);
		EXPECT_EQ(cryptanalysislib::uninitialized_copy(lsrc.begin(), lsrc.end(), p), p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); }
		cryptanalysislib::destroy(p, p + n);

		EXPECT_EQ(cryptanalysislib::uninitialized_copy_n(src.begin(), n, p), p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); }
		EXPECT_EQ(cryptanalysislib::destroy_n(p, n), p + n);

		auto moved = src;
		EXPECT_EQ(cryptanalysislib::uninitialized_move(moved.begin(), moved.end(), p), p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); ASSERT_TRUE(moved[i].empty()); }
		cryptanalysislib::destroy(p, p + n);

		moved = src;
		const auto r = cryptanalysislib::uninitialized_move_n(moved.begin(), n, p);
		EXPECT_EQ(r.first, moved.end());
		EXPECT_EQ(r.second, p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); }
		cryptanalysislib::destroy(p, p + n);

		cryptanalysislib::uninitialized_fill(p, p + n, str(3));
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], str(3)); }
		cryptanalysislib::destroy(p, p + n);

		EXPECT_EQ(cryptanalysislib::uninitialized_fill_n(p, n + 1, str(4)), p + n + 1);
		for (size_t i = 0; i <= n; ++i) { ASSERT_EQ(p[i], str(4)); }
		cryptanalysislib::destroy_n(p, n + 1);

		alloc.deallocate(p, n + 1);
	}
}

/// default and value construction
TEST(uninitialized, default_value_construct) {
	std::allocator<std::string> salloc;
	std::string *s = salloc.allocate(5);
	cryptanalysislib::uninitialized_default_construct(s, s + 5);
	for (size_t i = 0; i < 5; ++i) { EXPECT_TRUE(s[i].empty()); s[i] = str(i); }
	cryptanalysislib::destroy(s, s + 5);
	EXPECT_EQ(cryptanalysislib::uninitialized_default_construct_n(s, 5), s + 5);
	cryptanalysislib::destroy(s, s + 5);
	EXPECT_EQ(cryptanalysislib::uninitialized_value_construct_n(s, 5), s + 5);
	cryptanalysislib::destroy(s, s + 5);
	salloc.deallocate(s, 5);

	// value initialization zeroes integers
	alignas(uint64_t) unsigned char buf[8 * sizeof(uint64_t)];
	for (auto &b : buf) { b = 0xff; }
	uint64_t *u = reinterpret_cast<uint64_t *>(buf);
	cryptanalysislib::uninitialized_value_construct(u, u + 8);
	for (size_t i = 0; i < 8; ++i) { EXPECT_EQ(u[i], 0u); }
	cryptanalysislib::destroy(u, u + 8);
}

/// `construct_at` and `destroy_at`, also for arrays
TEST(uninitialized, construct_destroy_at) {
	std::allocator<std::string> alloc;
	std::string *p = alloc.allocate(1);
	std::string *q = cryptanalysislib::construct_at(p, 40, 'x');
	EXPECT_EQ(q, p);
	EXPECT_EQ(*p, std::string(40, 'x'));
	cryptanalysislib::destroy_at(p);
	alloc.deallocate(p, 1);

	using A = std::string[3];
	alignas(A) unsigned char raw[sizeof(A)];
	A *a = reinterpret_cast<A *>(raw);
	for (size_t i = 0; i < 3; ++i) { cryptanalysislib::construct_at(&(*a)[i], str(i)); }
	EXPECT_EQ((*a)[2], str(2));
	cryptanalysislib::destroy_at(a);
}

/// parallel versions: every element is constructed exactly once (values),
/// and destroyed exactly once (no leak/double free under ASan)
TEST(uninitialized, multithreaded) {
	const auto par = cryptanalysislib::par_if(true);
	std::allocator<std::string> alloc;
	for (const size_t n : {size_t(0), size_t(100), size_t(1u << 17u) + 7u}) {
		std::vector<std::string> src(n);
		for (size_t i = 0; i < n; ++i) { src[i] = str(i) + std::to_string(i); }
		std::string *p = alloc.allocate(n + 1);

		EXPECT_EQ(cryptanalysislib::uninitialized_copy(par, src.begin(), src.end(), p), p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); }
		cryptanalysislib::destroy(par, p, p + n);

		EXPECT_EQ(cryptanalysislib::uninitialized_copy_n(par, src.begin(), n, p), p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); }
		EXPECT_EQ(cryptanalysislib::destroy_n(par, p, n), p + n);

		auto moved = src;
		EXPECT_EQ(cryptanalysislib::uninitialized_move(par, moved.begin(), moved.end(), p), p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); ASSERT_TRUE(moved[i].empty()); }
		cryptanalysislib::destroy(par, p, p + n);

		moved = src;
		const auto r = cryptanalysislib::uninitialized_move_n(par, moved.begin(), n, p);
		EXPECT_EQ(r.first, moved.end());
		EXPECT_EQ(r.second, p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], src[i]); }
		cryptanalysislib::destroy(par, p, p + n);

		cryptanalysislib::uninitialized_fill(par, p, p + n, str(3));
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(p[i], str(3)); }
		cryptanalysislib::destroy(par, p, p + n);

		EXPECT_EQ(cryptanalysislib::uninitialized_fill_n(par, p, n + 1, str(4)), p + n + 1);
		for (size_t i = 0; i <= n; ++i) { ASSERT_EQ(p[i], str(4)); }
		cryptanalysislib::destroy_n(par, p, n + 1);

		cryptanalysislib::uninitialized_default_construct(par, p, p + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_TRUE(p[i].empty()); }
		cryptanalysislib::destroy(par, p, p + n);
		EXPECT_EQ(cryptanalysislib::uninitialized_default_construct_n(par, p, n), p + n);
		cryptanalysislib::destroy(par, p, p + n);
		EXPECT_EQ(cryptanalysislib::uninitialized_value_construct_n(par, p, n), p + n);
		cryptanalysislib::destroy(par, p, p + n);

		alloc.deallocate(p, n + 1);

		// value initialization zeroes integers
		std::vector<uint64_t> u(n + 1, 0xffu);
		cryptanalysislib::uninitialized_value_construct(par, u.begin(), u.begin() + n);
		for (size_t i = 0; i < n; ++i) { ASSERT_EQ(u[i], 0u); }
		EXPECT_EQ(u[n], 0xffu);
	}
}

/// usable in constant expressions (transient allocation)
TEST(uninitialized, constexpr_uninitialized) {
	constexpr auto r = []() {
		std::allocator<int> alloc;
		int *p = alloc.allocate(4);
		const int src[3] = {3, 4, 5};
		cryptanalysislib::uninitialized_copy(src, src + 3, p);
		cryptanalysislib::uninitialized_value_construct_n(p + 3, 1);
		const int s = p[0] * 1000 + p[1] * 100 + p[2] * 10 + p[3];
		cryptanalysislib::destroy(p, p + 4);
		alloc.deallocate(p, 4);
		return s;
	}();
	static_assert(r == 3450);
	EXPECT_EQ(r, 3450);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
