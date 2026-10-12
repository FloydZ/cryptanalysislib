#include <gtest/gtest.h>
#include <cstdint>

#include "jmp/jmp.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib::jmp;

TEST(jmp, array) {
	{
		array<int, 1u> a{42};
		EXPECT_EQ(1u, a.size());
		EXPECT_EQ(42, a[0u]);
	}

	{
		array<uint32_t, 2u> a{4u, 2u};
		EXPECT_EQ(2u, a.size());
		EXPECT_EQ(4u, a[0u]);
		EXPECT_EQ(2u, a[1u]);
	}
}

TEST(jmp, static_branch) {
	static_assert(1u == sizeof(static_branch<bool>));
	static_assert(1u == sizeof(static_branch<uint8_t , 0, 2>));
	static_assert(1u == sizeof(static_branch<uint16_t, 0, 3>));
	static_assert(1u == sizeof(static_branch<uint32_t, 1, 4>));
	static_assert(1u == sizeof(static_branch<uint64_t, 2, 7>));
	static_assert(1u == sizeof(static_branch<size_t, 1, 5>));

	static_assert(not [](auto... ts) { return requires { static_branch<bool>{ts...}; }; }());
	static_assert(not [](auto... ts) { return requires { static_branch<bool>{ts...}; }; }(static_branch<bool>{false}));
	static_assert([](auto value) { return requires { static_branch<bool>{value}; }; }(false));

	static_assert(not [](auto... ts) { return requires { static_branch<uint32_t, 0, 2>{ts...}; }; }());
	static_assert(not [](auto... ts) { return requires { static_branch<uint32_t, 0, 3>{ts...}; }; }(static_branch<uint32_t, 0, 3>{0u}));
	static_assert([](auto value) { return requires { static_branch<uint32_t, 1, 5>{value}; }; }(0u));
}

TEST(jmp, patching) {
	static constexpr static_branch<bool> b = false;

	auto fn_bool = [&] {
		if (b) {
			return 42;
		} else {
			return 0;
		}
	};

	static constexpr static_branch<int, 0, 2> i = 0;

	auto fn_int = [&] {
		switch (i) {
			default: return 0;
			case 0:  return 42;
			case 1:  return 99;
			case 2:  return 123;
		}
	};

	ASSERT_TRUE(init());

	EXPECT_EQ(0, fn_bool());
	b = false;
	EXPECT_EQ(0, fn_bool());
	b = true;
	EXPECT_EQ(42, fn_bool());
	b = false;
	EXPECT_EQ(0, fn_bool());

	EXPECT_EQ(42, fn_int());
	i = 0;
	EXPECT_EQ(42, fn_int());
	i = 1;
	EXPECT_EQ(99, fn_int());
	i = 2;
	EXPECT_EQ(123, fn_int());
	i = 0;
	EXPECT_EQ(42, fn_int());
}

/// every number of values (3..8) and a minimum != 0, every value in turn
template<int Min, int Max>
static void check_range() {
	static constexpr static_branch<int, Min, Max> s = Min;
	auto fn = [&]() -> int {
		const int v = s;
		return v;
	};
	ASSERT_TRUE(init());
	for (int round = 0; round < 2; ++round) {
		for (int v = Min; v <= Max; ++v) {
			s = v;
			EXPECT_EQ(fn(), v) << Min << " " << Max;
		}
	}
	s = Min;
}

TEST(jmp, ranges) {
	check_range<0, 2>();
	check_range<0, 3>();
	check_range<0, 4>();
	check_range<0, 5>();
	check_range<0, 6>();
	check_range<0, 7>();
	check_range<3, 10>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
