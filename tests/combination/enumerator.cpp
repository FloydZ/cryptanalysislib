#include <gtest/gtest.h>
#include <cstdint>
#include <set>
#include <vector>

#include "combination/bit_subset.h"
#include "combination/chase.h"
#include "combination/colex.h"
#include "combination/fibonacci_gray.h"
#include "combination/fibrep.h"
#include "combination/fibrep_subset_lexrev.h"
#include "combination/grey.h"
#include "combination/grsnegative.h"
#include "combination/kolaski.h"
#include "combination/lex.h"
#include "combination/lexicographic.h"
#include "combination/min_change.h"
#include "combination/necklace.h"
#include "combination/paren.h"
#include "combination/revbin_upd.h"
#include "combination/reverse_gray_code.h"
#include "combination/revolving_door.h"
#include "combination/rll2.h"
#include "combination/sequency.h"
#include "combination/shifts.h"
#include "combination/sl_gray.h"
#include "combination/subset_gray.h"
#include "math/math.h"

using ::testing::InitGoogleTest;

// every enumerator implements the common interface
static_assert(EnumeratorAble<bit_subset_T<uint64_t>>);
static_assert(EnumeratorAble<bit_subset_gray_T<uint64_t>>);
static_assert(EnumeratorAble<enumeration_colex<uint64_t, 10, 3>>);
static_assert(EnumeratorAble<bit_fibgray<uint64_t, 10>>);
static_assert(EnumeratorAble<fibrep>);
static_assert(EnumeratorAble<fibrev_subset_lexrev<uint64_t, 10>>);
static_assert(EnumeratorAble<gsr_negative>);
static_assert(EnumeratorAble<kolakoski_seq<uint64_t>>);
static_assert(EnumeratorAble<enumeration_lexicographic>);
static_assert(EnumeratorAble<BinaryLexicographic<uint32_t>>);
static_assert(EnumeratorAble<Combinations_Fq_Lexicographic>);
static_assert(EnumeratorAble<enumerate_t<10, 2>>);
static_assert(EnumeratorAble<enumeration_min_change<uint64_t>>);
static_assert(EnumeratorAble<bit_necklace<uint64_t, 10>>);
static_assert(EnumeratorAble<enumeration_parenthesis<uint64_t>>);
static_assert(EnumeratorAble<revbin_upd<uint64_t>>);
static_assert(EnumeratorAble<enumeration_reverse_gray_code<uint64_t>>);
static_assert(EnumeratorAble<combination_revdoor>);
static_assert(EnumeratorAble<bit_rll2<uint64_t>>);
static_assert(EnumeratorAble<sequence<uint64_t>>);
static_assert(EnumeratorAble<bit_comb_shifts<uint64_t, 10, 3>>);
static_assert(EnumeratorAble<bit_sl_gray>);
static_assert(EnumeratorAble<enumeration_gray<uint64_t>>);
static_assert(EnumeratorAble<Combinations_Binary_Chase<uint64_t, 10, 3>>);
static_assert(EnumeratorAble<Combinations_Fq_Chase<10, 3, 2>>);
static_assert(EnumeratorAble<chase_t<10, 2>>);
static_assert(EnumeratorAble<BinaryChaseEnumerator<10, 3>>);

/// \return all elements (via `key(current())`) from the current one on; checks
/// 	that `next()` returns false at the end and does not change the element
template<typename E, typename K>
static auto forward(E &e, K key, const size_t limit = 1u << 22) {
	std::vector<decltype(key(e.current()))> ret{key(e.current())};
	while (e.next()) {
		ret.push_back(key(e.current()));
		if (ret.size() > limit) { break; }
	}
	EXPECT_FALSE(e.next());
	EXPECT_EQ(key(e.current()), ret.back());
	return ret;
}

/// walks back from the last element of `seq` with `previous()`
template<typename E, typename K, typename V>
static void backward(E &e, K key, const V &seq) {
	for (size_t i = seq.size() - 1; i-- > 0;) {
		ASSERT_TRUE(e.previous()) << i;
		ASSERT_EQ(key(e.current()), seq[i]) << i;
	}
	EXPECT_FALSE(e.previous());
	EXPECT_EQ(key(e.current()), seq.front());
}

static auto id = [](const auto x) { return uint64_t(x); };

/// \return the bit mask of `k` positions
template<typename P>
static uint64_t mask(const P &pos, const uint32_t k) {
	uint64_t m = 0;
	for (uint32_t i = 0; i < k; ++i) { m |= 1ull << pos[i]; }
	return m;
}

/// \return true if all elements are distinct
template<typename V>
static bool distinct(const V &v) {
	std::set<typename V::value_type> s(v.begin(), v.end());
	return s.size() == v.size();
}

/// all words < 2^n with `k` set bits, ascending
static std::vector<uint64_t> combinations(const uint32_t n, const uint32_t k) {
	std::vector<uint64_t> ret;
	for (uint64_t x = 0; x < (1ull << n); ++x) {
		if (uint32_t(__builtin_popcountll(x)) == k) { ret.push_back(x); }
	}
	return ret;
}

TEST(enumerator, bit_subset) {
	for (const uint64_t V : {0x0ull, 0x1ull, 0b11010000100001ull, 0xf0f0ull}) {
		bit_subset_T<uint64_t> e(V);
		const auto s = forward(e, id);
		std::vector<uint64_t> ref;
		for (uint64_t x = 0; x <= V; ++x) { if ((x & ~V) == 0) { ref.push_back(x); } }
		EXPECT_EQ(s, ref);
		backward(e, id, s);
	}
}

TEST(enumerator, subset_gray) {
	for (const uint64_t V : {0x1ull, 0xbull, 0x1234ull}) {
		bit_subset_gray_T<uint64_t> e(V);
		const auto s = forward(e, id);
		EXPECT_EQ(s.size(), 1ull << __builtin_popcountll(V));
		EXPECT_TRUE(distinct(s));
		for (size_t i = 1; i < s.size(); ++i) {
			EXPECT_EQ(__builtin_popcountll(s[i] ^ s[i - 1]), 1);
			EXPECT_EQ(s[i] & ~V, 0u);
		}
		backward(e, id, s);
	}
}

template<typename T, const uint32_t n, const uint32_t k>
static void check_colex() {
	enumeration_colex<T, n, k> e;
	const auto s = forward(e, id);
	EXPECT_EQ(s, combinations(n, k)) << n << " " << k;
	backward(e, id, s);
}

TEST(enumerator, colex) {
	check_colex<uint64_t, 10, 3>();
	check_colex<uint64_t, 10, 0>();
	check_colex<uint64_t, 10, 10>();
	check_colex<uint64_t, 12, 5>();
	check_colex<uint32_t, 16, 4>();
	check_colex<uint8_t, 8, 3>();
}

template<const uint32_t n>
static void check_fibgray(const uint64_t fib) {
	bit_fibgray<uint64_t, n> e;
	const auto s = forward(e, id);
	EXPECT_EQ(s.size(), fib) << n;
	EXPECT_TRUE(distinct(s));
	for (size_t i = 0; i < s.size(); ++i) {
		EXPECT_EQ(s[i] & (s[i] >> 1u), 0u);
		EXPECT_LT(s[i], 1ull << n);
		if (i > 0) { EXPECT_EQ(__builtin_popcountll(s[i] ^ s[i - 1]), 1); }
	}
	backward(e, id, s);
}

TEST(enumerator, fibonacci_gray) {
	check_fibgray<1>(2);
	check_fibgray<5>(13);
	check_fibgray<10>(144);
	check_fibgray<16>(2584);
}

TEST(enumerator, fibrep) {
	fibrep e;
	std::vector<uint64_t> s{e.current()};
	for (uint64_t i = 1; i < 10000; ++i) {
		ASSERT_TRUE(e.next());
		ASSERT_EQ(fibrep::fibrep2bin(e.current()), i);
		ASSERT_EQ(e.current(), fibrep::bin2fibrep(i));
		ASSERT_TRUE(fibrep::is_fibrep(e.current()));
		s.push_back(e.current());
	}
	backward(e, id, s);
}

template<const uint32_t n>
static void check_fibrev(const uint64_t fib) {
	fibrev_subset_lexrev<uint64_t, n> e;
	const auto s = forward(e, id);
	EXPECT_EQ(s.size(), fib) << n;
	EXPECT_TRUE(distinct(s));
	EXPECT_EQ(s.front(), 1ull << (n - 1u));
	EXPECT_EQ(s.back(), 0u);
	for (const uint64_t x : s) {
		EXPECT_EQ(x & (x >> 1u), 0u);
		EXPECT_LT(x, 1ull << n);
	}
	backward(e, id, s);
}

TEST(enumerator, fibrep_subset_lexrev) {
	// the example of the docs (n = 6)
	fibrev_subset_lexrev<uint64_t, 6> e;
	const uint64_t ex[] = {0b100000, 0b101000, 0b101010, 0b101001, 0b100100, 0b100101,
	                       0b100010, 0b100001, 0b010000, 0b010100, 0b010101, 0b010010,
	                       0b010001, 0b001000, 0b001010, 0b001001, 0b000100, 0b000101,
	                       0b000010, 0b000001, 0b000000};
	for (size_t i = 0; i < 21; ++i) {
		EXPECT_EQ(e.current(), ex[i]) << i;
		EXPECT_EQ(e.next(), i != 20);
	}

	check_fibrev<1>(2);
	check_fibrev<6>(21);
	check_fibrev<12>(377);
}

TEST(enumerator, grs_negative) {
	gsr_negative e;
	std::vector<uint64_t> s{e.current()};
	for (uint64_t k = 1; k < 5000; ++k) {
		ASSERT_TRUE(e.next());
		ASSERT_EQ(e.index(), k);
		ASSERT_EQ(e.current(), gsr_negative::grs_negative_q(k));
		s.push_back(e.current());
	}
	backward(e, id, s);
}

TEST(enumerator, kolakoski) {
	// OEIS A000002
	const uint64_t a[] = {1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1,
	                      2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1};
	kolakoski_seq<uint64_t> e;
	for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); ++i) {
		EXPECT_EQ(e.current(), a[i]) << i;
		EXPECT_TRUE(e.next());
	}
}

TEST(enumerator, lexicographic) {
	for (uint32_t n = 1; n <= 12; ++n) {
		enumeration_lexicographic e(n);
		const auto s = forward(e, id);
		EXPECT_EQ(s.size(), 1ull << n);
		EXPECT_TRUE(distinct(s));
		EXPECT_EQ(s.back(), 0u);
		for (const uint64_t x : s) { EXPECT_LT(x, 1ull << n); }
		backward(e, id, s);

		BinaryLexicographic<uint32_t> f(n);
		const auto t = forward(f, id);
		EXPECT_EQ(t, s);
		backward(f, id, t);
	}
}

TEST(enumerator, fq_lexicographic) {
	for (const auto [n, q] : {std::pair{1u, 2u}, std::pair{3u, 3u}, std::pair{4u, 5u}, std::pair{6u, 2u}}) {
		Combinations_Fq_Lexicographic e(n, q);
		auto key = [](const std::vector<uint32_t> &v) { return v; };
		const auto s = forward(e, key);
		uint64_t N = 1;
		for (uint32_t i = 0; i < n; ++i) { N *= q; }
		ASSERT_EQ(s.size(), N);
		for (uint64_t i = 0; i < N; ++i) {
			// i in base q, the first coordinate is the most significant one
			uint64_t x = i;
			for (uint32_t j = n; j-- > 0;) {
				EXPECT_EQ(s[i][j], x % q);
				x /= q;
			}
		}
		backward(e, key, s);
	}
}

template<const uint32_t n, const uint32_t p>
static void check_enumerate_t() {
	enumerate_t<n, p> e;
	auto key = [](const uint16_t *c) { return mask(c, p); };
	const auto s = forward(e, key);
	// lexicographic order of the sorted positions = ascending order of the
	// reversed bit masks; compare as sets and check the order directly
	EXPECT_EQ(s.size(), bc(n, p));
	EXPECT_TRUE(distinct(s));
	enumerate_t<n, p> f;
	std::vector<std::vector<uint16_t>> lex;
	do { lex.emplace_back(f.current(), f.current() + p); } while (f.next());
	for (size_t i = 1; i < lex.size(); ++i) { EXPECT_LT(lex[i - 1], lex[i]); }
	backward(e, key, s);
}

TEST(enumerator, enumerate_t) {
	check_enumerate_t<10, 1>();
	check_enumerate_t<10, 2>();
	check_enumerate_t<12, 3>();
	check_enumerate_t<10, 4>();
}

TEST(enumerator, min_change) {
	for (uint32_t n = 1; n <= 12; ++n) {
		for (uint32_t k = 0; k <= n; ++k) {
			enumeration_min_change<uint64_t> e(n, k);
			const auto s = forward(e, id);
			EXPECT_EQ(s.size(), bc(n, k)) << n << " " << k;
			EXPECT_TRUE(distinct(s));
			backward(e, id, s);
		}
	}
}

TEST(enumerator, necklace) {
	bit_necklace<uint64_t, 12> e;
	const auto s = forward(e, id);
	// OEIS A000031
	EXPECT_EQ(s.size(), 352u);
	EXPECT_TRUE(distinct(s));
}

TEST(enumerator, parenthesis) {
	enumeration_parenthesis<uint64_t> e(6);
	const auto s = forward(e, id);
	// Catalan(6)
	EXPECT_EQ(s.size(), 132u);
	EXPECT_TRUE(distinct(s));
	backward(e, id, s);
}

TEST(enumerator, revbin_upd) {
	for (const uint32_t ldn : {1u, 2u, 5u, 10u}) {
		revbin_upd<uint64_t> e(1ull << (ldn - 1u));
		const auto s = forward(e, id);
		ASSERT_EQ(s.size(), 1ull << ldn);
		for (uint64_t k = 0; k < s.size(); ++k) {
			uint64_t r = 0;
			for (uint32_t b = 0; b < ldn; ++b) { r |= ((k >> b) & 1u) << (ldn - 1u - b); }
			EXPECT_EQ(s[k], r) << ldn << " " << k;
		}
		backward(e, id, s);
	}
}

template<typename T>
static void check_reverse_gray() {
	enumeration_reverse_gray_code<T> e;
	const auto s = forward(e, id);
	EXPECT_EQ(s.size(), sizeof(T) * 8u);
	EXPECT_TRUE(distinct(s));
	for (size_t i = 1; i < s.size(); ++i) {
		EXPECT_EQ(s[i], uint64_t(T(s[i - 1] ^ (s[i - 1] << 1u))));
	}
	// (1+x)^bits == 1
	EXPECT_EQ(T(s.back() ^ (s.back() << 1u)), T(1));
	backward(e, id, s);
}

TEST(enumerator, reverse_gray_code) {
	check_reverse_gray<uint8_t>();
	check_reverse_gray<uint16_t>();
	check_reverse_gray<uint32_t>();
	check_reverse_gray<uint64_t>();
}

TEST(enumerator, revolving_door) {
	combination_revdoor e(10, 4);
	auto key = [](const uint32_t *c) { return mask(c, 4); };
	const auto s = forward(e, key);
	EXPECT_EQ(s.size(), bc(10, 4));
	EXPECT_TRUE(distinct(s));
}

TEST(enumerator, rll2) {
	bit_rll2<uint8_t> e;
	const auto s = forward(e, id);
	EXPECT_TRUE(distinct(s));
	// F(8 + 2)
	EXPECT_EQ(s.size(), 55u);
	backward(e, id, s);
}

TEST(enumerator, sequency) {
	for (uint32_t n = 1; n <= 12; ++n) {
		for (uint32_t k = 0; k <= n; ++k) {
			sequence<uint64_t> e(k, n);
			const auto s = forward(e, id);
			std::set<uint64_t> ref;
			for (uint64_t x = 0; x < (1ull << n); ++x) {
				if (uint32_t(__builtin_popcountll(x ^ (x >> 1u))) == k) { ref.insert(x); }
			}
			EXPECT_EQ(std::set<uint64_t>(s.begin(), s.end()), ref) << n << " " << k;
			EXPECT_EQ(s.size(), ref.size());
			backward(e, id, s);
		}
	}
}

template<const uint32_t n, const uint32_t k>
static void check_shifts() {
	bit_comb_shifts<uint64_t, n, k> e;
	const auto s = forward(e, id);
	const auto ref = combinations(n, k);
	EXPECT_EQ(std::set<uint64_t>(s.begin(), s.end()), std::set<uint64_t>(ref.begin(), ref.end())) << n << " " << k;
	EXPECT_EQ(s.size(), ref.size());
}

TEST(enumerator, shifts) {
	check_shifts<6, 0>();
	check_shifts<6, 1>();
	check_shifts<6, 3>();
	check_shifts<10, 4>();
	check_shifts<12, 6>();
}

TEST(enumerator, sl_gray) {
	for (uint32_t n = 1; n <= 12; ++n) {
		bit_sl_gray e(n);
		const auto s = forward(e, id);
		EXPECT_EQ(s.size(), 1ull << n) << n;
		EXPECT_TRUE(distinct(s));
		for (size_t i = 1; i < s.size(); ++i) {
			EXPECT_EQ(__builtin_popcountll(s[i] ^ s[i - 1]), 1);
		}
	}
}

TEST(enumerator, gray) {
	for (const uint32_t n : {1u, 4u, 8u}) {
		enumeration_gray<uint8_t> e(n);
		const auto s = forward(e, id);
		EXPECT_EQ(s.size(), 1ull << n);
		EXPECT_TRUE(distinct(s));
		for (size_t i = 1; i < s.size(); ++i) {
			EXPECT_EQ(__builtin_popcountll(s[i] ^ s[i - 1]), 1);
		}
		backward(e, id, s);
	}
}

template<const uint32_t n, const uint32_t w>
static void check_binary_chase() {
	Combinations_Binary_Chase<uint64_t, n, w> e;
	auto key = [](const auto &a) { return a[0]; };
	const auto s = forward(e, key);
	EXPECT_EQ(s.size(), bc(n, w)) << n << " " << w;
	EXPECT_TRUE(distinct(s));
	for (const uint64_t x : s) {
		EXPECT_EQ(uint32_t(__builtin_popcountll(x)), w);
		EXPECT_LT(x, 1ull << n);
	}
}

TEST(enumerator, binary_chase) {
	check_binary_chase<6, 0>();
	check_binary_chase<6, 1>();
	check_binary_chase<10, 3>();
	check_binary_chase<12, 4>();
}

template<const uint32_t n, const uint32_t q, const uint32_t w>
static void check_fq_chase() {
	Combinations_Fq_Chase<n, q, w> e;
	auto key = [](const auto &a) { return std::vector<uint32_t>(a.begin(), a.end()); };
	const auto s = forward(e, key);
	uint64_t N = bc(n, w);
	for (uint32_t i = 0; i < w; ++i) { N *= q - 1u; }
	EXPECT_EQ(s.size(), N) << n << " " << q << " " << w;
	EXPECT_TRUE(distinct(s));
	for (const auto &v : s) {
		uint32_t wt = 0;
		for (const uint32_t x : v) { EXPECT_LT(x, q); wt += x != 0; }
		EXPECT_EQ(wt, w);
	}
}

TEST(enumerator, fq_chase) {
	check_fq_chase<6, 2, 2>();
	check_fq_chase<6, 3, 2>();
	check_fq_chase<8, 5, 3>();
	check_fq_chase<5, 4, 1>();
}

template<const uint32_t n, const uint32_t p>
static void check_chase_t() {
	chase_t<n, p> e;
	auto key = [](const uint16_t *c) { return mask(c, p); };
	const auto s = forward(e, key);
	EXPECT_EQ(s.size(), bc(n, p));
	EXPECT_TRUE(distinct(s));

	// the same order as `enumerate`: f(added, removed)
	chase_t<n, p> f;
	uint64_t x = (1ull << p) - 1u;
	size_t i = 1;
	f.enumerate([&](const uint16_t a, const uint16_t r) {
		x ^= (1ull << a) ^ (1ull << r);
		if (i < s.size()) { EXPECT_EQ(s[i], x) << n << " " << p << " " << i; }
		i += 1;
	});
	EXPECT_EQ(i, s.size());
	backward(e, key, s);
}

TEST(enumerator, chase_t) {
	check_chase_t<10, 1>();
	check_chase_t<10, 2>();
	check_chase_t<11, 2>();
	check_chase_t<10, 3>();
	check_chase_t<13, 3>();
}

template<const uint32_t n, const uint32_t t>
static void check_chase_enumerator() {
	BinaryChaseEnumerator<n, t> e;
	auto key = [](const auto &c) { return mask(c, t); };
	const auto s = forward(e, key);
	EXPECT_EQ(s.size(), bc(n, t));
	EXPECT_TRUE(distinct(s));
	backward(e, key, s);
}

TEST(enumerator, binary_chase_enumerator) {
	check_chase_enumerator<6, 1>();
	check_chase_enumerator<8, 3>();
	check_chase_enumerator<12, 4>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
