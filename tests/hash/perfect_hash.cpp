#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "hash/perfect_hash.h"

using ::testing::InitGoogleTest;
namespace mph = cryptanalysislib::mph;
using namespace std::literals;

// the self tests of https://github.com/qlibs/mph
TEST(perfect_hash, utility) {
	// compressed_pair
	static_assert(1 == mph::utility::compressed_pair{1, 2}.first);
	static_assert(2 == mph::utility::compressed_pair{1, 2}.second);
	static_assert(sizeof(int) + sizeof(int) == sizeof(mph::utility::compressed_pair{int{}, int{}}));
	struct empty { };
	static_assert(sizeof(int) == sizeof(mph::utility::compressed_pair{empty{}, 42}));
	static_assert(sizeof(int) == sizeof(mph::utility::compressed_pair{42, empty{}}));
	static_assert(sizeof(empty) + sizeof(empty) == sizeof(mph::utility::compressed_pair{empty{}, empty{}}));

	// array
	{
		mph::utility::array<mph::u32, 1> a{};
		EXPECT_EQ(1u, a.size());
	}
	{
		mph::utility::array a{1, 2};
		EXPECT_EQ(2u, a.size());
		EXPECT_EQ(1, a[0]);
		EXPECT_EQ(2, a[1]);
	}
	{
		mph::utility::array a{1};
		a[0] = 2;
		EXPECT_EQ(2, a[0]);
	}
	{
		mph::utility::array a{1, 2, 3};
		EXPECT_EQ(3u, a.size());
		EXPECT_NE(a.begin(), a.end());
		EXPECT_EQ(a.size(), mph::u32(a.end() - a.begin()));
		EXPECT_EQ(a.end(), a.begin() + a.size());
	}
}

TEST(perfect_hash, type_traits) {
	static_assert(0 == mph::type_traits::constant<0>::value);
	static_assert(0 == mph::type_traits::constant_v<0>);
	static_assert(42 == mph::type_traits::constant<42>{});
	static_assert(42 == mph::type_traits::constant_v<42>);
	static_assert(42u == mph::type_traits::constant<42u>::value);
	static_assert('X' == mph::type_traits::constant<'X'>::value);
	static_assert('X' == mph::type_traits::constant_v<'X'>);

	using mph::u32;
	using mph::u64;
	static_assert(sizeof(u32) == sizeof(mph::type_traits::conditional_t<true, u32, u64>));
	static_assert(sizeof(u64) == sizeof(mph::type_traits::conditional_t<false, u32, u64>));
}

TEST(perfect_hash, random) {
	mph::random::pcg random{};
	EXPECT_NE(random(), random());
}

TEST(perfect_hash, pext_mask) {
	using mph::u32;
	using mph::u64;
	using mph::utility::array;
	using mph::utility::compressed_pair;

	static_assert(0    == mph::detail::pext(0b00, 0b00));
	static_assert(0    == mph::detail::pext(0b01, 0b00));
	static_assert(0b1  == mph::detail::pext(0b01, 0b01));
	static_assert(0b01 == mph::detail::pext(0b01, 0b11));
	static_assert(0b0  == mph::detail::pext(0b01, 0b10));
	static_assert(0b1  == mph::detail::pext(0b11, 0b10));
	static_assert(0b1  == mph::detail::pext(0b11, 0b01));
	static_assert(0b11 == mph::detail::pext(0b11, 0b11));

	// the same at run time (`pext` instruction with BMI2)
	for (u32 a = 0; a < 4; ++a) {
		for (u32 m = 0; m < 4; ++m) {
			u32 e = 0, k = 0;
			for (u32 i = 0; i < 2; ++i) {
				if ((m >> i) & 1u) { e |= ((a >> i) & 1u) << k++; }
			}
			EXPECT_EQ(e, mph::detail::pext(a, m));
		}
	}

	static_assert(0b01 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b10, 0}, compressed_pair{0b11, 0}}));
	static_assert(0b10 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b01, 0}, compressed_pair{0b11, 0}}));
	static_assert(0b10 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b11, 0}, compressed_pair{0b01, 0}}));
	static_assert(0b01 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b11, 0}, compressed_pair{0b10, 0}}));
	static_assert(0b01 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b01, 0}, compressed_pair{0b10, 0}}));
	static_assert(0b01 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b10, 0}, compressed_pair{0b01, 0}}));
	static_assert(0b11 == mph::detail::mask<u32, 1u>(array{compressed_pair{0b11, 0}, compressed_pair{0b11, 0}}));

	static_assert(sizeof(mph::detail::narrow_mask<array{compressed_pair{u32{1}, 0}, compressed_pair{u32{1}<<1, 0}}>())==sizeof(u32));
	static_assert(sizeof(mph::detail::narrow_mask<array{compressed_pair{u64{1}, 0}, compressed_pair{u64{1}<<32, 0}}>())==sizeof(u32));
	static_assert(sizeof(mph::detail::narrow_mask<array{compressed_pair{u64{1}<<32, 0}, compressed_pair{u64{1}<<33, 0}}>())==sizeof(u64));
}

TEST(perfect_hash, traits) {
	using mph::u8;
	using mph::u32;
	using mph::utility::array;
	using mph::type_traits::constant;
	{
		constexpr const auto& entries = constant<array{u32(4)}>::value;
		constexpr const auto v = mph::detail::traits<entries>::mappings;
		static_assert(1u == v.size());
		static_assert(u32(4) == v[0].first);
		static_assert(u8(0) == v[0].second);
	}
	{
		constexpr const auto& entries = constant<array{u32(4), u32(2)}>::value;
		constexpr const auto v = mph::detail::traits<entries>::mappings;
		static_assert(2u == v.size());
		static_assert(u32(4) == v[0].first);
		static_assert(u32(2) == v[1].first);
		static_assert(u8(0) == v[0].second);
		static_assert(u8(1) == v[1].second);
	}
}

TEST(perfect_hash, conditional_optional_to) {
	static_assert(0 == mph::conditional<0u>(true, 1, 0));
	static_assert(0 == mph::conditional<0u>(false, 1, 0));
	static_assert(1 == mph::conditional<50u>(true, 1, 0));
	static_assert(0 == mph::conditional<50u>(false, 1, 0));
	static_assert(1 == mph::conditional<100u>(true, 1, 0));
	static_assert(1 == mph::conditional<100u>(false, 1, 0));

	using mph::u32;
	static_assert(not [](auto value) { return requires { mph::optional<u32>(value); }; }(""));
	static_assert([](auto value) { return requires { mph::optional<u32>(value); }; }(u32{}));
	static_assert(not mph::optional<u32>{});
	static_assert(not mph::optional<u32>{u32{}, false});
	static_assert(mph::optional<u32>{42});
	static_assert(42 == *mph::optional<u32>{42});
	static_assert(mph::optional<bool>{true});
	static_assert(*mph::optional<bool>{true});
	static_assert(mph::optional<bool>{false});
	static_assert(not *mph::optional<bool>{false});

	using mph::utility::array;
	static_assert(0 == mph::to<u32>(0));
	static_assert(42 == mph::to<u32>(42));
	static_assert(42u == mph::to<u32>(42u));
	static_assert(0 == mph::to<u32>(""));
	static_assert((u32('A') << 0) + (u32('B') << __CHAR_BIT__) == mph::to<u32>("AB"));
	static_assert((u32('A') << 0) + (u32('B') << __CHAR_BIT__) + (u32('C') << __CHAR_BIT__*2) == mph::to<u32>("ABC"));
	static_assert((u32('A') << 0) + (u32('B') << __CHAR_BIT__) == mph::to<u32>(array{'A','B'}));
	static_assert((u32('A') << 0) + (u32('B') << __CHAR_BIT__) + (u32('C') << __CHAR_BIT__*2) == mph::to<u32>(array{'A','B','C'}));
	// run time
	EXPECT_EQ((u32('A') << 0) + (u32('B') << __CHAR_BIT__), mph::to<u32>("AB"sv));
}

TEST(perfect_hash, lookup) {
	using mph::u16;
	using mph::u32;
	using mph::u64;
	using mph::to;
	using mph::utility::compressed_pair;
	using mph::utility::array;
	using mph::type_traits::constant;
	{
		constexpr const auto& entries = constant<array{compressed_pair{u32(4), u32(2)}}>::value;
		auto lookup = mph::lookup<entries>;
		static_assert(2 == lookup(u32(4)));
	}
	{
		constexpr const auto& entries = constant<[] {
			std::array<int, 512> a{};
			auto count = 0;
			for (auto &v : a) v = count++;
			return a;
		}()>::value;
		static_assert(0 == mph::lookup<entries>(u16(0)));
		static_assert(1 == mph::lookup<entries>(u16(1)));
		static_assert(128 == mph::lookup<entries>(u16(128)));
		static_assert(255 == mph::lookup<entries>(u16(255)));
		static_assert(256 == mph::lookup<entries>(u16(256)));
		static_assert(entries.size()-1 == mph::lookup<entries>(u16(entries.size()-1)));
		for (u32 i = 0; i < entries.size(); ++i) {
			EXPECT_EQ(i, mph::lookup<entries>(u16(i)));
		}
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{u32(4), u32(2)},
			compressed_pair{u32(42), u32(87)},
			compressed_pair{u32(100), u32(100)},
		}>::value;
		static_assert(2 == mph::lookup<entries>(u32(4)));
		static_assert(87 == mph::lookup<entries>(u32(42)));
		static_assert(100 == mph::lookup<entries>(u32(100)));
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{u64{4}<<32, u32(2)},
			compressed_pair{u64{42}<<32, u32(87)},
			compressed_pair{u64{100}<<32, u32(100)},
		}>::value;
		static_assert(2 == mph::lookup_pext<entries>{}(u64{4}<<32));
		static_assert(87 == mph::lookup<entries>(u64{42}<<32));
		static_assert(100 == mph::lookup<entries>(u64{100}<<32));
	}
	{
		enum class color { red, green, blue };
		constexpr const auto& entries = constant<array{
			compressed_pair{color::red, 0},
			compressed_pair{color::green, 1},
			compressed_pair{color::blue, 2},
		}>::value;
		static_assert(0 == mph::lookup<entries>(color::red));
		static_assert(1 == mph::lookup<entries>(color::green));
		static_assert(2 == mph::lookup<entries>(color::blue));
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{to<u32>("BTC"), 1},
			compressed_pair{to<u32>("ETH"), 2},
			compressed_pair{to<u32>("XRP"), 3},
		}>::value;
		static_assert(1 == mph::lookup<entries>("BTC"));
		static_assert(2 == mph::lookup<entries>("ETH"));
		static_assert(3 == mph::lookup<entries>("XRP"));
	}
}

TEST(perfect_hash, find) {
	using mph::u32;
	using mph::u64;
	using mph::to;
	using mph::utility::compressed_pair;
	using mph::utility::array;
	using mph::type_traits::constant;
	{
		constexpr const auto& entries = constant<array{compressed_pair{u32(4), u32(2)}}>::value;
		auto find = mph::find<entries>;
		static_assert(not find(u32(0)));
		static_assert(not find(u32(1)));
		static_assert(find(u32(4)));
		static_assert(0 == *find(u32(0)));
		static_assert(0 == *find(u32(1)));
		static_assert(2 == *find(u32(4)));
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{u32(4), u32(2)},
			compressed_pair{u32(42), u32(87)},
			compressed_pair{u32(100), u32(100)},
		}>::value;
		static_assert(not mph::find<entries>(u32(0)));
		static_assert(mph::find<entries>(u32(4)));
		static_assert(mph::find<entries>(u32(42)));
		static_assert(mph::find<entries>(u32(100)));
		static_assert(0 == *mph::find<entries>(u32(0)));
		static_assert(2 == *mph::find<entries>(u32(4)));
		static_assert(87 == *mph::find<entries>(u32(42)));
		static_assert(100 == *mph::find<entries>(u32(100)));
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{u64(4)<<32, u32(2)},
			compressed_pair{u64(42)<<32, u32(87)},
			compressed_pair{u64(100)<<32, u32(100)},
		}>::value;
		static_assert(not mph::find<entries>(u64(0)));
		static_assert(mph::find<entries>(u64(4)<<32));
		static_assert(mph::find<entries>(u64(42)<<32));
		static_assert(mph::find<entries>(u64(100)<<32));
		static_assert(0 == *mph::find<entries>(u64(0)));
		static_assert(2 == *mph::find<entries>(u64(4)<<32));
		static_assert(87 == *mph::find<entries>(u64(42)<<32));
		static_assert(100 == *mph::find<entries>(u64(100)<<32));
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{u32(9), u32(0)},
			compressed_pair{u32(6), u32(1)},
			compressed_pair{u32(14), u32(2)},
			compressed_pair{u32(11), u32(3)},
		}>::value;
		static_assert(not mph::find<entries>(u32(42)));
		static_assert(0 == *mph::find<entries>(u32(9)));
		static_assert(1 == *mph::find<entries>(u32(6)));
		static_assert(2 == *mph::find<entries>(u32(14)));
		static_assert(3 == *mph::find<entries>(u32(11)));
	}
	{
		enum class color { red, green, blue };
		constexpr const auto& entries = constant<array{
			compressed_pair{color::red, 0},
			compressed_pair{color::green, 1},
			compressed_pair{color::blue, 2},
		}>::value;
		static_assert(mph::find<entries>(color::red));
		static_assert(mph::find<entries>(color::green));
		static_assert(mph::find<entries>(color::blue));
		static_assert(0 == *mph::find<entries>(color::red));
		static_assert(1 == *mph::find<entries>(color::green));
		static_assert(2 == *mph::find<entries>(color::blue));
	}
	{
		constexpr const auto& entries = constant<array{
			compressed_pair{to<u32>("BTC"), 1},
			compressed_pair{to<u32>("ETH"), 2},
			compressed_pair{to<u32>("XRP"), 3},
		}>::value;
		static_assert(not mph::find<entries>(""));
		static_assert(mph::find<entries>("BTC"));
		static_assert(mph::find<entries>("ETH"));
		static_assert(mph::find<entries>("XRP"));
		static_assert(0 == *mph::find<entries>(""));
		static_assert(1 == *mph::find<entries>("BTC"));
		static_assert(2 == *mph::find<entries>("ETH"));
		static_assert(3 == *mph::find<entries>("XRP"));
	}
}

/// N pseudo random distinct keys (compile time), value = index
template<typename T, const uint32_t N>
constexpr auto random_entries() {
	std::array<std::pair<T, uint32_t>, N> a{};
	uint64_t s = 0x9e3779b97f4a7c15ull;
	for (uint32_t i = 0; i < N; ++i) {
		while (true) {
			s ^= s << 13u; s ^= s >> 7u; s ^= s << 17u;
			const T k = T(s);
			bool dup = (k == 0);
			for (uint32_t j = 0; j < i; ++j) { dup |= (a[j].first == k); }
			if (!dup) { a[i] = {k, i}; break; }
		}
	}
	return a;
}

constexpr static auto entries_u32_64  = random_entries<uint32_t, 64>();
constexpr static auto entries_u64_64  = random_entries<uint64_t, 64>();
constexpr static auto entries_u32_200 = random_entries<uint32_t, 200>();

/// `find` at run time (`pext`, <= 64 keys, and SIMD buckets, > 64 keys)
/// against `std::unordered_map`, for the keys and for random other keys
template<const auto &entries>
static void check_find() {
	using T = typename std::remove_cvref_t<decltype(entries)>::value_type::first_type;
	std::unordered_map<T, uint32_t> m;
	for (const auto &[k, v] : entries) { m[k] = v; }

	for (const auto &[k, v] : entries) {
		const auto r = mph::find<entries>(k);
		ASSERT_TRUE(r) << k;
		EXPECT_EQ(*r, v);
		EXPECT_EQ(mph::lookup<entries>(k), v);
	}

	uint64_t s = 42;
	for (uint32_t i = 0; i < 100000; ++i) {
		s = s * 6364136223846793005ull + 1442695040888963407ull;
		const T k = T(s >> 11u);
		const auto r = mph::find<entries>(k);
		const auto it = m.find(k);
		ASSERT_EQ(bool(r), it != m.end()) << k;
		if (r) { EXPECT_EQ(*r, it->second); }
	}
}

TEST(perfect_hash, find_random_keys) {
	check_find<entries_u32_64>();
	check_find<entries_u64_64>();
#ifdef CRYPTANALYSISLIB_MPH_SIMD
	// more than 64 keys need `<experimental/simd>` (not in libc++)
	check_find<entries_u32_200>();
	// empty slots of the SIMD buckets
	EXPECT_FALSE(mph::find<entries_u32_200>(uint32_t(0)));
#endif
}

/// strings of 5..8 chars (64 bit keys), whose distinguishing bits are in
/// the first 4 chars (32 bit mask)
TEST(perfect_hash, long_strings) {
	constexpr static auto symbols = std::array{
		"AAAAXYZ"sv, "BBBBXYZ"sv, "CCCCXYZ"sv, "DDDDXYZ"sv, "EEEEEXYZ"sv,
	};
	for (uint32_t i = 0; i < symbols.size(); ++i) {
		const auto r = mph::find<symbols>(symbols[i]);
		ASSERT_TRUE(r) << symbols[i];
		EXPECT_EQ(*r, i);
	}
	EXPECT_FALSE(mph::find<symbols>("AAAAXYW"sv));
}

TEST(perfect_hash, strings) {
	constexpr static auto symbols = std::array{
		"BTC "sv, "ETH "sv, "BNB "sv, "SOL "sv, "XRP "sv, "DOGE"sv,
		"TON "sv, "ADA "sv, "SHIB"sv, "AVAX"sv, "LINK"sv, "BCH "sv,
	};
	for (uint32_t i = 0; i < symbols.size(); ++i) {
		const auto r = mph::find<symbols>(symbols[i]);
		ASSERT_TRUE(r);
		EXPECT_EQ(*r, i);
	}
	EXPECT_FALSE(mph::find<symbols>("ABCD"sv));
	EXPECT_FALSE(mph::find<symbols>("BTC"sv));

	enum class color { red, green, blue };
	constexpr static auto colors = std::array{
		std::pair{"red"sv, color::red},
		std::pair{"green"sv, color::green},
		std::pair{"blue"sv, color::blue},
	};
	static_assert(color::green == mph::lookup<colors>("green"));
	static_assert(not mph::find<colors>("unknown"));
	EXPECT_EQ(*mph::find<colors>("blue"sv), color::blue);
	EXPECT_FALSE(mph::find<colors>("gray"sv));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
