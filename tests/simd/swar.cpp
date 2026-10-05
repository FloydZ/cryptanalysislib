#include <gtest/gtest.h>
#include "simd/simd.h"

TEST(swar, simple) {
	constexpr auto expect = [](bool cond) { if (not cond) { void failed(); failed(); } };

	using u8 = uint8_t;
	using u16 = uint16_t;
	using u32 = uint32_t;
	using u64 = uint64_t;
    using namespace cryptanalysislib;

	// swar::broadcast
	{
		static_assert(u8(0x01) == swar::broadcast<u8, 8u>(0x01));
		static_assert(u16(0x01'01) == swar::broadcast<u16, 8u>(0x01));
		static_assert(u32(0x42'42'42'42) == swar::broadcast<u32, 8u>(0x42));
		static_assert(u16(0b0111'0111'0111'0111) == swar::broadcast<u16, 4u>(0b111));
		static_assert(u64(0x1234'1234'1234'1234) == swar::broadcast<u64, 16u>(0x1234));
		static_assert(u64(0x12345678'12345678) == swar::broadcast<u64, 32u>(0x12345678));
	}

	// swar::swar
	{
		{
			constexpr swar::swar<u8> s{};
			static_assert(8u == s.size());
		}

		{
			constexpr swar::swar<u16> s{};
			static_assert(4u == s.size());
		}

		{
			constexpr swar::swar<u32> s{};
			static_assert(2u == s.size());
		}

		{
			constexpr swar::swar<u8, 1u> s{};
			static_assert(1u == s.size());
		}

		{
			swar::swar<u8, 1u> s{};
			expect(1u == s.size());
		}

		{
			swar::swar<u8, 4u> s{42};
			expect(4u == s.size());
			expect(42 == s[0u]);
			expect(42 == s[1u]);
			expect(42 == s[2u]);
			expect(42 == s[3u]);
		}

		{
			const u8 data[]{1, 2, 3, 4};
			swar::swar<u8, 4u> s{data};
			expect(4u == s.size());
			expect(1 == s[0u]);
			expect(2 == s[1u]);
			expect(3 == s[2u]);
			expect(4 == s[3u]);
		}

		{
			swar::swar<u8, 8u> s{[](auto i) { return i * 2; }};
			expect(8u == s.size());
			expect(0 * 2 == s[0u]);
			expect(1 * 2 == s[1u]);
			expect(2 * 2 == s[2u]);
			expect(3 * 2 == s[3u]);
			expect(4 * 2 == s[4u]);
			expect(5 * 2 == s[5u]);
			expect(6 * 2 == s[6u]);
			expect(7 * 2 == s[7u]);
		}

		{
			swar::swar<u32, 2u> lhs{1};
			swar::swar<u32, 2u> rhs = lhs;
			expect(u64(lhs) == u64(rhs));
		}

		{
			swar::swar<u32, 2u> lhs{1};
			swar::swar<u32, 2u> rhs = static_cast<swar::swar<u32, 2u> &&>(lhs);
			expect(u64(lhs) == u64(rhs));
		}

		{
			{
				swar::swar<u8, 4u> s{};
				expect(not decltype(s)::abi_type(s));
			}

			{
				swar::swar<u8, 4u> s{1};
				expect(decltype(s)::abi_type(s));
			}
		}
	}

	// swar::swar_mask
	{
		{
			constexpr swar::swar_mask<u8, 8u> s{};
			static_assert(8u == s.size());
		}

		{
			const u16 data1[]{1, 2, 3, 4};
			const swar::swar<u16, 4u> lhs{data1};
			const u16 data2[]{1, 3, 3, 1};
			const swar::swar<u16, 4u> rhs{data2};

			const swar::swar_mask<u16, 4u> sm = lhs == rhs;

			expect(sm[0u]);
			expect(not sm[1u]);
			expect(sm[2u]);
			expect(not sm[3u]);
		}

		{
			const u8 data1[]{1, 2, 3, 4, 5, 6, 7, 8};
			const swar::swar<u8, 8u> lhs{data1};
			const u8 data2[]{1, 2, 3, 4, 0, 6, 7, 8};
			const swar::swar<u8, 8u> rhs{data2};

			const swar::swar_mask<u8, 8u> sm = lhs == rhs;

			expect(sm[0u]);
			expect(sm[1u]);
			expect(sm[2u]);
			expect(sm[3u]);
			expect(not sm[4u]);
			expect(sm[5u]);
			expect(sm[6u]);
			expect(sm[7u]);
		}
	}

	// swar::all_of
	{
		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{4};
			expect(swar::all_of(lhs == rhs));
		}

		{
			const swar::swar<u16, 4u> lhs{123};
			const swar::swar<u16, 4u> rhs{123};
			expect(swar::all_of(lhs == rhs));
		}

		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{7};
			expect(not swar::all_of(lhs == rhs));
		}

		{
			const u8 data[]{1, 2, 3, 4};
			const swar::swar<u8, 4u> lhs{data};
			expect(not swar::all_of(lhs == swar::swar<u8, 4u>{}));
			expect(not swar::all_of(lhs == swar::swar<u8, 4u>{1}));
			expect(not swar::all_of(lhs == swar::swar<u8, 4u>{2}));
			expect(not swar::all_of(lhs == swar::swar<u8, 4u>{3}));
			expect(not swar::all_of(lhs == swar::swar<u8, 4u>{4}));
		}
	}

	// swar::any_of
	{
		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{4};
			expect(swar::any_of(lhs == rhs));
		}

		{
			const swar::swar<u16, 4u> lhs{123};
			const swar::swar<u16, 4u> rhs{123};
			expect(swar::any_of(lhs == rhs));
		}

		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{7};
			expect(not swar::any_of(lhs == rhs));
		}

		{
			const u8 data[]{1, 2, 3, 4};
			const swar::swar<u8, 4u> lhs{data};
			expect(not swar::any_of(lhs == swar::swar<u8, 4u>{}));
			expect(not swar::any_of(lhs == swar::swar<u8, 4u>{42}));
			expect(swar::any_of(lhs == swar::swar<u8, 4u>{1}));
			expect(swar::any_of(lhs == swar::swar<u8, 4u>{2}));
			expect(swar::any_of(lhs == swar::swar<u8, 4u>{3}));
			expect(swar::any_of(lhs == swar::swar<u8, 4u>{4}));
		}
	}

	// swar::some_of
	{
		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{4};
			expect(not swar::some_of(lhs == rhs));
		}

		{
			const swar::swar<u16, 4u> lhs{123};
			const swar::swar<u16, 4u> rhs{123};
			expect(not swar::some_of(lhs == rhs));
		}

		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{7};
			expect(not swar::some_of(lhs == rhs));
		}

		{
			const u8 data[]{1, 2, 3, 4};
			const swar::swar<u8, 4u> lhs{data};
			expect(not swar::some_of(lhs == swar::swar<u8, 4u>{}));
			expect(not swar::some_of(lhs == swar::swar<u8, 4u>{42}));
			expect(swar::some_of(lhs == swar::swar<u8, 4u>{1}));
			expect(swar::some_of(lhs == swar::swar<u8, 4u>{2}));
			expect(swar::some_of(lhs == swar::swar<u8, 4u>{3}));
			expect(swar::some_of(lhs == swar::swar<u8, 4u>{4}));
		}
	}

	// swar::none_of
	{
		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{4};
			expect(not swar::none_of(lhs == rhs));
		}

		{
			const swar::swar<u16, 4u> lhs{123};
			const swar::swar<u16, 4u> rhs{123};
			expect(not swar::none_of(lhs == rhs));
		}

		{
			const swar::swar<u8, 4u> lhs{4};
			const swar::swar<u8, 4u> rhs{7};
			expect(swar::none_of(lhs == rhs));
		}

		{
			const u8 data[]{1, 2, 3, 4};
			const swar::swar<u8, 4u> lhs{data};
			expect(swar::none_of(lhs == swar::swar<u8, 4u>{}));
			expect(swar::none_of(lhs == swar::swar<u8, 4u>{42}));
			expect(not swar::none_of(lhs == swar::swar<u8, 4u>{1}));
			expect(not swar::none_of(lhs == swar::swar<u8, 4u>{2}));
			expect(not swar::none_of(lhs == swar::swar<u8, 4u>{3}));
			expect(not swar::none_of(lhs == swar::swar<u8, 4u>{4}));
		}
	}

	// swar::find_first_set
	{
		{
			const u8 data[]{1, 2, 3, 4};
			const swar::swar<u8, 4u> lhs{data};
			expect(0u == swar::find_first_set(lhs == swar::swar<u8, 4u>{1}));
			expect(1u == swar::find_first_set(lhs == swar::swar<u8, 4u>{2}));
			expect(2u == swar::find_first_set(lhs == swar::swar<u8, 4u>{3}));
			expect(3u == swar::find_first_set(lhs == swar::swar<u8, 4u>{4}));
		}

		{
			const u32 data[]{1234, 5678};
			const swar::swar<u32, 2u> lhs{data};
			expect(0u == swar::find_first_set(lhs == swar::swar<u32, 2u>{1234}));
			expect(1u == swar::find_first_set(lhs == swar::swar<u32, 2u>{5678}));
		}

		{
			const u8 data[]{5, 6, 7, 8, 5, 6, 1, 1};
			const swar::swar<u8, 8u> lhs{data};
			expect(0u == swar::find_first_set(lhs == swar::swar<u8, 8u>{5}));
			expect(1u == swar::find_first_set(lhs == swar::swar<u8, 8u>{6}));
			expect(2u == swar::find_first_set(lhs == swar::swar<u8, 8u>{7}));
			expect(3u == swar::find_first_set(lhs == swar::swar<u8, 8u>{8}));
			expect(0u == swar::find_first_set(lhs == swar::swar<u8, 8u>{5}));
			expect(1u == swar::find_first_set(lhs == swar::swar<u8, 8u>{6}));
			expect(6u == swar::find_first_set(lhs == swar::swar<u8, 8u>{1}));
			expect(6u == swar::find_first_set(lhs == swar::swar<u8, 8u>{1}));
		}
	}

	// swar::find_last_set
	{
		{
			const u8 data[]{1, 2, 3, 4};
			const swar::swar<u8, 4u> lhs{data};
			expect(0u == swar::find_last_set(lhs == swar::swar<u8, 4u>{1}));
			expect(1u == swar::find_last_set(lhs == swar::swar<u8, 4u>{2}));
			expect(2u == swar::find_last_set(lhs == swar::swar<u8, 4u>{3}));
			expect(3u == swar::find_last_set(lhs == swar::swar<u8, 4u>{4}));
		}

		{
			const u32 data[]{1234, 5678};
			const swar::swar<u32, 2u> lhs{data};
			expect(0u == swar::find_last_set(lhs == swar::swar<u32, 2u>{1234}));
			expect(1u == swar::find_last_set(lhs == swar::swar<u32, 2u>{5678}));
		}

		{
			const u8 data[]{5, 6, 7, 8, 5, 6, 1, 1};
			const swar::swar<u8, 8u> lhs{data};
			expect(4u == swar::find_last_set(lhs == swar::swar<u8, 8u>{5}));
			expect(5u == swar::find_last_set(lhs == swar::swar<u8, 8u>{6}));
			expect(2u == swar::find_last_set(lhs == swar::swar<u8, 8u>{7}));
			expect(3u == swar::find_last_set(lhs == swar::swar<u8, 8u>{8}));
			expect(4u == swar::find_last_set(lhs == swar::swar<u8, 8u>{5}));
			expect(5u == swar::find_last_set(lhs == swar::swar<u8, 8u>{6}));
			expect(7u == swar::find_last_set(lhs == swar::swar<u8, 8u>{1}));
			expect(7u == swar::find_last_set(lhs == swar::swar<u8, 8u>{1}));
		}
	}

	// swar::popcount
	{
		expect(0u == popcount(swar::swar_mask<u8, 4>{}));
		expect(1u * 4u == popcount(swar::swar_mask<u8, 4>{0x01'01'01'01}));
		expect(3u * 4u == popcount(swar::swar_mask<u8, 4>{0b00000111'00000111'00000111'00000111}));
	}

	// swar::is_swar_v
	{
		static_assert(not swar::is_swar_v<void>);
		static_assert(not swar::is_swar_v<int>);
		static_assert(not swar::is_swar_v<swar::swar_mask<u8, 2u>>);
		static_assert(not swar::is_swar_v<swar::swar_mask<u32, 2u>>);
		static_assert(swar::is_swar_v<swar::swar<u8, 2u>>);
		static_assert(swar::is_swar_v<swar::swar<u32, 2u>>);
	}

	// swar::is_swar_mask_v
	{
		static_assert(not swar::is_swar_mask_v<void>);
		static_assert(not swar::is_swar_mask_v<int>);
		static_assert(not swar::is_swar_mask_v<swar::swar<u8, 2u>>);
		static_assert(not swar::is_swar_mask_v<swar::swar<u32, 2u>>);
		static_assert(swar::is_swar_mask_v<swar::swar_mask<u8, 2u>>);
		static_assert(swar::is_swar_mask_v<swar::swar_mask<u32, 2u>>);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
