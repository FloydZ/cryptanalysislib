#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "reflection/reflection.h"
#include "thread/performance.h"

using ::testing::InitGoogleTest;

#ifdef CRYPTANALYSISLIB_REFLECTIONS_AVAILABLE
using namespace reflect;

/// the self tests of https://github.com/qlibs/reflect (v1.2.5)
namespace reflect_test {
	struct empty { };
	struct foo { using type = int; enum E { }; };
	template<class T> struct optional {
		constexpr optional() = default;
		constexpr optional(T t) : t{t} { }
		T t{};
	};
	struct bar { };
	template<class T> struct foo_t { };
	namespace ns::inline v1 { template<auto...> struct bar_v { }; } // ns
	enum e {
		_141 = 141,
		_142 = 142,
	};
	enum e1 { e1a };
	enum class e2 : std::uint8_t { e2a, e2b };
	enum e3 { e3a, e3b, e3c, };
	enum class e4 : std::int32_t { e4a, e4b = 5, e4c, e4d = 2, };
	consteval auto enum_min(e) { return e::_141; }
	consteval auto enum_max(e) { return e::_142; }
} // namespace reflect_test

// enum_name probes values in [REFLECT_ENUM_MIN, REFLECT_ENUM_MAX]; clang >= 20 rejects
// casting such values to an enum without a fixed underlying type
enum E : int { A, B };
struct foo { int a; E b; };

TEST(simple, my_type) {
	constexpr auto f = foo{.a = 42, .b = B};

	// reflect::size
	static_assert(2 == reflect::size(f));

	// reflect::type_id
	static_assert(reflect::type_id(f.a) != reflect::type_id(f.b));

	// reflect::type_name
	static_assert("foo" == reflect::type_name(f));
	static_assert("int" == reflect::type_name(f.a));
	static_assert("E"   == reflect::type_name(f.b));

	// reflect::enum_name
	static_assert("B" == reflect::enum_name(f.b));

	// reflect::member_name
	static_assert("a" == reflect::member_name<0>(f));
	static_assert("b" == reflect::member_name<1>(f));

	// reflect::get
	static_assert(42 == reflect::get<0>(f)); // by index
	static_assert(B  == reflect::get<1>(f));

	static_assert(42 == reflect::get<"a">(f)); // by name
	static_assert(B  == reflect::get<"b">(f));

	// reflect::to
	// NOTE: not `constexpr`, as the tuple holds references to the members of `f`
	const auto t = reflect::to<std::tuple>(f);
	EXPECT_EQ(42, std::get<0>(t));
	EXPECT_EQ(B, std::get<1>(t));

	reflect::for_each([](auto) {
		std::cout << reflect::type_name(E{});
	}, f);

	reflect::for_each([f](auto I) {
		std::cout << reflect::type_name(f)                         // foo, foo
		          << ", " << reflect::member_name<I>(f)             // a  , b
		          << ", " << reflect::type_name(reflect::get<I>(f)) // int, E
		          << ", " << reflect::get<I>(f)                     // 42 , B
		          << ", " << reflect::size_of<I>(f)                 // 4  , 4
		          << ", " << reflect::align_of<I>(f)                // 4  , 4
		          << ", " << reflect::offset_of<I>(f) << std::endl; // 0  , 4
	}, f);
}

TEST(simple, nth_pack_element) {
	using reflect::detail::nth_pack_element;

	static_assert(1 == nth_pack_element<0>(1));
	static_assert(1 == nth_pack_element<0>(1, 2));
	static_assert(2 == nth_pack_element<1>(1, 2));
	static_assert('a' == nth_pack_element<0>('a', 1, true));
	static_assert(1 == nth_pack_element<1>('a', 1, true));
	static_assert(true == nth_pack_element<2>('a', 1, true));

	{
		[[maybe_unused]] int a{};
		static_assert(std::is_same_v<int&, decltype(nth_pack_element<0>(a))>);
	}

	{
		[[maybe_unused]] const int a{};
		static_assert(std::is_same_v<const int&, decltype(nth_pack_element<0>(a))>);
	}

	{
		[[maybe_unused]] int a{};
		static_assert(std::is_same_v<int&&, decltype(nth_pack_element<0>(static_cast<int&&>(a)))>);
	}

	{
		[[maybe_unused]] const int a{};
		static_assert(std::is_same_v<const int&&, decltype(nth_pack_element<0>(static_cast<const int&&>(a)))>);
	}
}

TEST(simple, fixed_string) {
	static_assert(0u == std::size(fixed_string{""}));
	static_assert(fixed_string{""} == fixed_string{""});
	static_assert(std::string_view{""} == std::string_view{fixed_string{""}});
	static_assert(3u == std::size(fixed_string{"foo"}));
	static_assert(std::string_view{"foo"} == std::string_view{fixed_string{"foo"}});
	static_assert(fixed_string{"foo"} == fixed_string{"foo"});
}

TEST(simple, size) {
	using reflect_test::empty;
	struct one { int a; };
	struct two { int a; reflect_test::optional<int> o; };
	struct empty_with_base : empty {};
	struct one_with_base : empty { int a; };
	struct two_with_base : empty { int a; reflect_test::optional<int> o; };
	struct base {};
	struct empty_with_bases : empty, base {};
	struct one_with_bases : empty, base { int a; };
	struct two_with_bases : empty, base { int a; reflect_test::optional<int> o; };
	struct two_with_base_member : empty, base { int a; base b; };
	using tuple_with_values = std::tuple<int, int, int>;
	using tuple_with_refs = std::tuple<int const&, int &, int *>;
	using pair_test = std::pair<float, std::string_view>;

	static_assert(0 == size<empty>());
	static_assert(0 == size(empty{}));
	static_assert(1 == size<one>());
	static_assert(1 == size(one{}));
	static_assert(2 == size<two>());
	static_assert(2 == size(two{}));
	static_assert(0 == size<empty_with_base>());
	static_assert(0 == size(empty_with_base{}));
	static_assert(1 == size<one_with_base>());
	static_assert(1 == size(one_with_base{}));
	static_assert(2 == size<two_with_base>());
	static_assert(2 == size(two_with_base{}));
	static_assert(0 == size<empty_with_bases>());
	static_assert(0 == size(empty_with_bases{}));
	static_assert(1 == size<one_with_bases>());
	static_assert(1 == size(one_with_bases{}));
	static_assert(2 == size<two_with_bases>());
	static_assert(2 == size(two_with_bases{}));
	static_assert(2 == size<two_with_base_member>());
	static_assert(2 == size(two_with_base_member{}));
	static_assert(0 == size<const empty>());
	static_assert(1 == size<const one>());
	static_assert(2 == size<const two>());
	static_assert(0 == size<const empty_with_base>());
	static_assert(1 == size<const one_with_base>());
	static_assert(2 == size<const two_with_base>());
	static_assert(0 == size<const empty_with_bases>());
	static_assert(1 == size<const one_with_bases>());
	static_assert(2 == size<const two_with_bases>());
	static_assert(2 == size<const two_with_base_member>());
	static_assert(3 == reflect::size<tuple_with_values>());
	static_assert(3 == reflect::size<tuple_with_refs>());
	static_assert(2 == reflect::size<pair_test>());

	struct non_standard_layout {
	private:
		[[maybe_unused]] int _1{};
	public:
		[[maybe_unused]] int _2{};
	};

	struct S {
		double _1;
		non_standard_layout _2;
		float _3;
	};

	static_assert(3 == size<S>());

	constexpr auto test = []<class T> {
		struct S { T _1; T _2; int _3; T _4; };
		struct S5_0 { int _1; int _2; int _3; int _4; T _5; };
		struct S5_1 { T _1; int _2; int _3; int _4; int _5; };
		struct S5_2 { int _1; int _2; T _3; int _4; int _5; };
		struct S5_3 { int _1; int _2; T _3; int _4; T _5; };
		struct S5_4 { T _1; T _2; T _3; T _4; T _5; };
		struct S6 { T _1; T _2; T _3; T _4; T _5;  T _6;};

		static_assert(4 == size<S>());
		static_assert(5 == size<S5_0>());
		static_assert(5 == size<S5_1>());
		static_assert(5 == size<S5_2>());
		static_assert(5 == size<S5_3>());
		static_assert(5 == size<S5_4>());
		static_assert(6 == size<S6>());
	};

	{
		struct T {
			T() = default;
			T(T&&) = default;
			T(const T&) = delete;
			T& operator=(T&&) = default;
			T& operator=(const T&) = delete;
		};
		test.template operator()<T>();
	}

	{
		struct T {
			T(T&&) = default;
			T(const T&) = delete;
			T& operator=(T&&) = default;
			T& operator=(const T&) = delete;
		};
		test.template operator()<T>();
	}

	{
		struct T { T(int) { } };
		test.template operator()<T>();
	}

	struct bf {
		unsigned int _1: 1;
		unsigned int _2: 1;
		unsigned int _3: 1;
		unsigned int _4: 1;
		unsigned int _5: 1;
		unsigned int _6: 1;
	};

	static_assert(6 == size<bf>());

	struct {
		int _1;
		int _2;
	} anonymous;

	static_assert(2 == size(anonymous));
}

TEST(simple, visit) {
	using reflect_test::empty;
	static_assert(0 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, empty{}));

	struct one { int a; };
	static_assert(1 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, one{}));

	struct two { int a; int b; };
	static_assert(2 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, two{}));

	using ts = std::tuple<int,float,double>;
	static_assert(3 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, ts{}));

	// the largest supported number of members
	struct s64 {
		int _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,
		    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32,
		    _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48,
		    _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64;
	};
	static_assert(64 == visit([]([[maybe_unused]] auto&&... args) { return sizeof...(args); }, s64{}));
	static_assert(std::string_view{"_64"} == member_name<63, s64>());
}

TEST(simple, type_name) {
	struct local {};

	static_assert(std::string_view{"void"} == type_name<void>());
	static_assert(std::string_view{"int"} == type_name<int>());
	static_assert(std::string_view{"empty"} == type_name<reflect_test::empty>());
	static_assert(std::string_view{"empty"} == type_name(reflect_test::empty{}));
	static_assert(std::string_view{"foo"} == type_name<reflect_test::foo>());
	static_assert(std::string_view{"foo"} == type_name(reflect_test::foo{}));
	static_assert(std::string_view{"bar"} == type_name<reflect_test::bar>());
	static_assert(std::string_view{"bar"} == type_name(reflect_test::bar{}));
	static_assert(std::string_view{"foo_t"} == type_name<reflect_test::foo_t<void>>());
	static_assert(std::string_view{"foo_t"} == type_name<reflect_test::foo_t<int>>());
	static_assert(std::string_view{"foo_t"} == type_name<reflect_test::foo_t<reflect_test::ns::bar_v<42>>>());
	static_assert(std::string_view{"bar_v"} == type_name<reflect_test::ns::bar_v<42>>());
	static_assert(std::string_view{"bar_v"} == type_name<reflect_test::ns::bar_v<>>());
	static_assert(std::string_view{"int"} == type_name(reflect_test::foo::type{}));
	static_assert(std::string_view{"E"} == type_name(reflect_test::foo::E{}));
	static_assert(std::string_view{"local"} == type_name<local>());
	static_assert(std::string_view{"local"} == type_name(local{}));
	static_assert(std::string_view{"tuple"} == type_name(std::tuple<int,int>{}));

	// the run time version
	EXPECT_EQ(std::string_view{"local"}, type_name<local>());
}

TEST(simple, type_id) {
	static_assert(type_id<reflect_test::bar>() != type_id(reflect_test::foo{}));
	static_assert(type_id<int>() != type_id(reflect_test::foo{}));
	static_assert(type_id<int>() != type_id<void>());
	static_assert(type_id<void>() != type_id<int>());
	static_assert(type_id<void>() == type_id<void>());
	static_assert(type_id<int>() == type_id<int>());
	static_assert(type_id<int>() == type_id<int&>());
	static_assert(type_id<const int&>() == type_id<int&>());
	static_assert(type_id<void*>() == type_id<void>());
	static_assert(type_id<reflect_test::foo>() == type_id(reflect_test::foo{}));
	static_assert(type_id<reflect_test::bar>() == type_id(reflect_test::bar{}));

	static_assert(type_id_v<void> == type_id_v<void>);
	static_assert(type_id_v<int>  != type_id_v<void>);
	static_assert(type_id_v<int>  == type_id_v<int>);
	static_assert(type_id_v<int>  != type_id_v<const int&>);
}

TEST(simple, enumerators) {
	static_assert(1u == enumerators<reflect_test::e1, 0, 1>.size());
	static_assert(0u == enumerators<reflect_test::e1, 0, 1>[0].first);

	static_assert(2u == enumerators<reflect_test::e2, 0, 2>.size());
	static_assert(std::uint8_t(0) == enumerators<reflect_test::e2, 0, 2>[0].first);
	static_assert(std::uint8_t(1) == enumerators<reflect_test::e2, 0, 2>[1].first);

	static_assert(3u == enumerators<reflect_test::e3, 0, 3>.size());
	static_assert(0u == enumerators<reflect_test::e3, 0, 3>[0].first);
	static_assert(1u == enumerators<reflect_test::e3, 0, 3>[1].first);
	static_assert(2u == enumerators<reflect_test::e3, 0, 3>[2].first);

	static_assert(4u == enumerators<reflect_test::e4, 0, 6>.size());
	static_assert(0 == enumerators<reflect_test::e4, 0, 6>[0].first);
	static_assert(2 == enumerators<reflect_test::e4, 0, 6>[1].first);
	static_assert(5 == enumerators<reflect_test::e4, 0, 6>[2].first);
	static_assert(6 == enumerators<reflect_test::e4, 0, 6>[3].first);
}

TEST(simple, enum_name) {
	enum class foobar {
		foo = 1, bar = 2
	};

	enum mask : unsigned char {
		a = 0b00,
		b = 0b01,
		c = 0b10,
	};

	enum sparse {
		_128 = 128,
		_130 = 130,
	};

	enum class negative {
		unknown = -1,
		A = 0,
		B = 1,
	};

	static_assert([](auto e) { return requires { enum_name<foobar,"",1,2>(e); }; }(foobar::foo));
	static_assert([](auto e) { return requires { enum_name<mask,"",1,2>(e); }; }(mask::a));
	static_assert(not [](auto e) { return requires { enum_name<foobar,"",1,2>(e); }; }(0));
	static_assert(not [](auto e) { return requires { enum_name<int,"",1,2>(e); }; }(0));
	static_assert(not [](auto e) { return requires { enum_name<int,"",1,2>(e); }; }(42u));

	static_assert(std::string_view{""} == enum_name<foobar,"",1,2>(static_cast<foobar>(42)));
	static_assert(std::string_view{"unknown"} == enum_name<foobar,"unknown",1,2>(static_cast<foobar>(42)));

	const auto e = foobar::foo;
	static_assert(std::string_view{"foo"} == enum_name<foobar,"",1,2>(e));

	static_assert(std::string_view{"foo"} == enum_name<foobar,"",1,2>(foobar::foo));
	static_assert(std::string_view{"bar"} == enum_name<foobar,"",1,2>(foobar::bar));

	static_assert(std::string_view{"a"} == enum_name<mask,"",0,2>(mask::a));
	static_assert(std::string_view{"b"} == enum_name<mask,"",0,2>(mask::b));
	static_assert(std::string_view{"c"} == enum_name<mask,"",0,2>(mask::c));

	static_assert(std::string_view{"_128"} == enum_name<sparse,"",128,130>(sparse::_128));
	static_assert(std::string_view{"_130"} == enum_name<sparse,"",128,130>(sparse::_130));

	static_assert(std::string_view{"unknown"} == enum_name<negative,"<>",-1, 1>(negative::unknown));
	static_assert(std::string_view{"A"} == enum_name<negative,"<>",-1, 1>(negative::A));
	static_assert(std::string_view{"B"} == enum_name<negative,"<>",-1, 1>(negative::B));
	static_assert(std::string_view{"<>"} == enum_name<negative,"<>",-1, 1>(static_cast<negative>(-42)));

	// `enum_min`/`enum_max` overloads (found via ADL)
	static_assert(std::string_view{"_141"} == enum_name<reflect_test::e>(reflect_test::e::_141));
	static_assert(std::string_view{"_142"} == enum_name<reflect_test::e>(reflect_test::e::_142));

	// the default range [REFLECT_ENUM_MIN, REFLECT_ENUM_MAX], and the run time version
	EXPECT_EQ(std::string_view{"unknown"}, enum_name(negative::unknown));
	EXPECT_EQ(std::string_view{"B"}, enum_name(negative::B));
	EXPECT_EQ(std::string_view{""}, enum_name(static_cast<negative>(-42)));
}

TEST(simple, member_name) {
	struct foo { int i; bool b; void* bar{}; };

	static_assert(std::string_view{"i"} == member_name<0, foo>());
	static_assert(std::string_view{"i"} == member_name<0>(foo{}));

	static_assert(std::string_view{"b"} == member_name<1, foo>());
	static_assert(std::string_view{"b"} == member_name<1>(foo{}));

	static_assert(std::string_view{"bar"} == member_name<2, foo>());
	static_assert(std::string_view{"bar"} == member_name<2>(foo{}));

	// the run time version
	EXPECT_EQ(std::string_view{"bar"}, member_name<2>(foo{}));
}

TEST(simple, get_by_index) {
	struct foo { int i; bool b; };

	{
		constexpr auto f = foo{.i=42, .b=true};

		static_assert([]<auto N> { return requires { get<N>(f); }; }.template operator()<0>());
		static_assert([]<auto N> { return requires { get<N>(f); }; }.template operator()<1>());
		static_assert(not []<auto N> { return requires { get<N>(f); }; }.template operator()<2>());

		static_assert(42 == get<0>(f));
		static_assert(true == get<1>(f));
	}

	{
		{
			auto f = foo{};
			auto value = get<0>(f);
			static_assert(std::is_same_v<decltype(value), int>);
		}

		{
			auto value = get<0>(foo{});
			static_assert(std::is_same_v<decltype(value), int>);
		}

		{
			auto f = foo{};
			auto& lvalue = get<0>(f);
			static_assert(std::is_same_v<decltype(lvalue), int&>);
			lvalue = 5;
			EXPECT_EQ(5, f.i);
		}

		{
			static_assert(std::is_same_v<decltype(get<0>(foo{})), int&&>);
		}
	}

	{
		using tuple_foo = std::tuple<int, bool>;
		auto f = tuple_foo{};
		[[maybe_unused]] auto value0 = get<0>(f);
		static_assert(std::is_same_v<decltype(value0), int>);

		[[maybe_unused]] auto value1 = get<1>(f);
		static_assert(std::is_same_v<decltype(value1), bool>);
	}
}

TEST(simple, member_type) {
	struct foo{ int i; float j; bool k; };

	static_assert(std::is_same_v<int, member_type<0, foo>>);
	static_assert(std::is_same_v<float, member_type<1, foo>>);
	static_assert(std::is_same_v<bool, member_type<2, foo>>);

	using tuple_foo = std::tuple<int, bool>;

	static_assert(std::is_same_v<int, member_type<0, tuple_foo>>);
	static_assert(std::is_same_v<bool, member_type<1, tuple_foo>>);
}

TEST(simple, has_member_name) {
	struct foo { int bar; };
	static_assert(has_member_name<foo, "bar">);
	static_assert(not has_member_name<foo, "baz">);
	static_assert(not has_member_name<foo, "BAR">);
	static_assert(not has_member_name<foo, "">);
}

TEST(simple, diagnose_member_name) {
	struct foo { bool flag; int bar; };
	static_assert(detail::diagnose_member_name<foo, "">() == "`foo` has no data member named ``. Did you mean `bar`?");
	static_assert(detail::diagnose_member_name<foo, "ba">() == "`foo` has no data member named `ba`. Did you mean `bar`?");
	static_assert(detail::diagnose_member_name<foo, "fl">() == "`foo` has no data member named `fl`. Did you mean `flag`?");
	static_assert(detail::diagnose_member_name<foo, "baz">() == "`foo` has no data member named `baz`. Did you mean `bar`?");
	static_assert(detail::diagnose_member_name<foo, "lag">() == "`foo` has no data member named `lag`. Did you mean `flag`?");

	using reflect_test::empty;
	static_assert(detail::diagnose_member_name<empty, "any">() == "`empty` has no data members.");
}

TEST(simple, index_of) {
	struct foo { int a; int b; };
	static_assert(0 == index_of<"a", foo>());
	static_assert(1 == index_of<"b">(foo{}));
}

TEST(simple, get_by_name) {
	struct foo { int i; bool b; };

	{
		constexpr auto f = foo{.i=42, .b=true};

		static_assert([]<fixed_string Name> { return requires { get<Name>(f); }; }.template operator()<"i">());
		static_assert([]<fixed_string Name> { return requires { get<Name>(f); }; }.template operator()<"b">());
		static_assert(not []<fixed_string Name> { return requires { get<Name>(f); }; }.template operator()<"unknown">());

		static_assert(42 == get<"i">(f));
		static_assert(true == get<"b">(f));
	}

	{
		auto f = foo{};
		auto value = get<"i">(f);
		static_assert(std::is_same_v<decltype(value), int>);
	}

	{
		auto value = get<"i">(foo{});
		static_assert(std::is_same_v<decltype(value), int>);
	}

	{
		auto f = foo{};
		auto& lvalue = get<"i">(f);
		static_assert(std::is_same_v<decltype(lvalue), int&>);
	}

	{
		static_assert(std::is_same_v<decltype(get<"i">(foo{})), int&&>);
	}
}

TEST(simple, to_tuple) {
	struct foo { int a; int b; };

	{
		constexpr auto t = to<std::tuple>(foo{.a=4, .b=2});
		static_assert(4 == std::get<0>(t));
		static_assert(2 == std::get<1>(t));
	}

	{
		auto f = foo{.a=4, .b=2};
		auto t = to<std::tuple>(f);
		std::get<0>(t) *= 10;
		f.b = 42;
		EXPECT_TRUE(40 == std::get<0>(t) and 40 == f.a);
		EXPECT_TRUE(42 == std::get<1>(t) and 42 == f.b);
	}

	{
		const auto f = foo{.a=4, .b=2};
		auto t = to<std::tuple>(f);
		EXPECT_EQ(f.a, std::get<0>(t));
		EXPECT_EQ(f.b, std::get<1>(t));
	}
}

TEST(simple, copy) {
	struct foo {
		int a{};
		int b{};
	};

	struct bar {
		int a{};
		int b{};
	};

	const auto f = foo{.a=1, .b=2};

	{
		bar b{};
		b.b = 42;
		copy<"a">(f, b);
		EXPECT_EQ(b.a, f.a);
		EXPECT_EQ(42, b.b);
	}

	{
		bar b{};
		b.b = 42;
		copy<"a", "b">(f, b);
		EXPECT_EQ(b.a, f.a);
		EXPECT_EQ(b.b, f.b);
	}

	{
		bar b{};
		b.a = 42;
		copy<"b">(f, b);
		EXPECT_EQ(42, b.a);
		EXPECT_EQ(b.b, f.b);
	}

	{
		bar b{};
		copy<"c">(f, b); // ignores
		EXPECT_EQ(0, b.a);
		EXPECT_EQ(0, b.b);
	}

	{
		bar b{};
		copy<"a", "c", "b">(f, b); // copies a, b; ignores c
		EXPECT_EQ(b.a, f.a);
		EXPECT_EQ(b.b, f.b);
	}

	{
		bar b{};
		copy(f, b);
		EXPECT_EQ(b.a, f.a);
		EXPECT_EQ(b.b, f.b);
	}

	{
		bar b{.a=4, .b=2};
		copy(f, b); // overwrites members
		EXPECT_EQ(b.a, f.a);
		EXPECT_EQ(b.b, f.b);
	}

	struct baz {
		int a{};
		int c{};
	};

	{
		baz b{};
		b.c = 42;
		copy(f, b);
		EXPECT_EQ(b.a, f.a);
		EXPECT_EQ(42, b.c);
	}
}

TEST(simple, to_struct) {
	struct foo {
		int a{};
		int b{};
	};

	struct bar {
		int a{};
		int b{};
	};

	{
		constexpr auto b = to<bar>(foo{.a=4, .b=2});
		static_assert(4 == b.a);
		static_assert(2 == b.b);
	}

	{
		auto f = foo{.a=4, .b=2};
		auto b = to<bar>(f);
		f.a = 42;
		EXPECT_EQ(42, f.a);
		EXPECT_EQ(4, b.a);
		EXPECT_EQ(2, b.b);
	}

	{
		const auto f = foo{.a=4, .b=2};
		const auto b = to<bar>(f);
		EXPECT_EQ(4, b.a);
		EXPECT_EQ(2, b.b);
	}

	struct baz {
		int a{};
		int c{};
	};

	{
		auto f = foo{.a=4, .b=2};
		auto b = to<bar>(f);
		b.a = 1;
		EXPECT_TRUE(1 == b.a and 4 == f.a);
		EXPECT_TRUE(2 == b.b and 2 == f.b);
	}

	{
		const auto b = to<baz>(foo{.a=4, .b=2});
		EXPECT_TRUE(4 == b.a and 0 == b.c);
	}

	struct foobar {
		int a{};
		enum e : int { } b; // strong type, type conversion disabled
	};

	{
		const auto fb = to<foobar>(foo{.a=4, .b=2});
		EXPECT_TRUE(4 == fb.a and 0 == fb.b);
	}
}

TEST(simple, size_of) {
	struct s {
		float a;
		char b;
		char bb;
		int c;
	};

	static_assert(sizeof(float) == size_of<0, s>());
	static_assert(sizeof(char) == size_of<1, s>());
	static_assert(sizeof(char) == size_of<2, s>());
	static_assert(sizeof(int) == size_of<3, s>());
	static_assert(sizeof(float) == size_of<"a", s>());
	static_assert(sizeof(char) == size_of<"b", s>());
	static_assert(sizeof(char) == size_of<"bb", s>());
	static_assert(sizeof(int) == size_of<"c", s>());
}

TEST(simple, align_of) {
	struct s {
		float a;
		char b;
		char bb;
		int c;
	};

	static_assert(alignof(float) == align_of<0, s>());
	static_assert(alignof(char) == align_of<1, s>());
	static_assert(alignof(char) == align_of<2, s>());
	static_assert(alignof(int) == align_of<3, s>());
	static_assert(alignof(float) == align_of<"a", s>());
	static_assert(alignof(char) == align_of<"b", s>());
	static_assert(alignof(char) == align_of<"bb", s>());
	static_assert(alignof(int) == align_of<"c", s>());
}

TEST(simple, offset_of) {
	struct s {
		float a;
		char b;
		char bb;
		int c;
	};

#pragma pack(push, 1)
	struct s2 {
		float a;
		char b;
		char bb;
		int c;
		double d;
		char e;
	};
#pragma pack(pop)

	struct a {
		int i;
		int j;
	};

	struct b {
		int i;
		int k;
	};

	struct al {
		char a;
		int b;
		char c;
	};

	using reflect_test::empty;

	struct p {
		int i;
		[[no_unique_address]] empty e;
	};

	static_assert(offset_of<0, a>() == __builtin_offsetof(a, i));
	static_assert(offset_of<1, a>() == __builtin_offsetof(a, j));
	static_assert(offset_of<"i", a>() == __builtin_offsetof(a, i));
	static_assert(offset_of<"j", a>() == __builtin_offsetof(a, j));
	static_assert(offset_of<0, b>() == __builtin_offsetof(b, i));
	static_assert(offset_of<1, b>() == __builtin_offsetof(b, k));
	static_assert(offset_of<"i", b>() == __builtin_offsetof(b, i));
	static_assert(offset_of<"k", b>() == __builtin_offsetof(b, k));
	static_assert(offset_of<0, s>() == __builtin_offsetof(s, a));
	static_assert(offset_of<1, s>() == __builtin_offsetof(s, b));
	static_assert(offset_of<2, s>() == __builtin_offsetof(s, bb));
	static_assert(offset_of<3, s>() == __builtin_offsetof(s, c));
	static_assert(offset_of<"a", s>() == __builtin_offsetof(s, a));
	static_assert(offset_of<"b", s>() == __builtin_offsetof(s, b));
	static_assert(offset_of<"bb", s>() == __builtin_offsetof(s, bb));
	static_assert(offset_of<"c", s>() == __builtin_offsetof(s, c));
	static_assert(offset_of<0, s2>() == __builtin_offsetof(s2, a));
	static_assert(offset_of<1, s2>() == __builtin_offsetof(s2, b));
	static_assert(offset_of<2, s2>() == __builtin_offsetof(s2, bb));
	static_assert(offset_of<3, s2>() == __builtin_offsetof(s2, c));
	static_assert(offset_of<4, s2>() == __builtin_offsetof(s2, d));
	static_assert(offset_of<5, s2>() == __builtin_offsetof(s2, e));
	static_assert(offset_of<"a", s2>() == __builtin_offsetof(s2, a));
	static_assert(offset_of<"b", s2>() == __builtin_offsetof(s2, b));
	static_assert(offset_of<"bb", s2>() == __builtin_offsetof(s2, bb));
	static_assert(offset_of<"c", s2>() == __builtin_offsetof(s2, c));
	static_assert(offset_of<"d", s2>() == __builtin_offsetof(s2, d));
	static_assert(offset_of<"e", s2>() == __builtin_offsetof(s2, e));
	static_assert(offset_of<0, al>() == __builtin_offsetof(al, a));
	static_assert(offset_of<1, al>() == __builtin_offsetof(al, b));
	static_assert(offset_of<2, al>() == __builtin_offsetof(al, c));
	static_assert(offset_of<"a", al>() == __builtin_offsetof(al, a));
	static_assert(offset_of<"b", al>() == __builtin_offsetof(al, b));
	static_assert(offset_of<"c", al>() == __builtin_offsetof(al, c));
	static_assert(offset_of<1, p>() == __builtin_offsetof(p, e));
	static_assert(offset_of<"e", p>() == __builtin_offsetof(p, e));
}

TEST(simple, for_each) {
	struct foo { short _1; int _2; } f{._1=1, ._2=2};

	std::array<std::size_t, reflect::size<foo>()> size_of{};
	std::array<std::size_t, reflect::size<foo>()> align_of{};
	std::array<std::size_t, reflect::size<foo>()> offset_of{};
	std::array<std::string_view, reflect::size<foo>()> name{};
	std::array<std::string_view, reflect::size<foo>()> type{};
	std::array<std::common_type_t<short, int>, reflect::size<foo>()> value{};

	auto i = 0;
	for_each([&](auto I) {
		size_of[i] = reflect::size_of<I>(f);
		align_of[i] = reflect::align_of<I>(f);
		offset_of[i] = reflect::offset_of<I>(f);
		name[i] = reflect::member_name<I>(f);
		type[i] = reflect::type_name(reflect::get<I>(f));
		value[i] = reflect::get<I>(f);
		++i;
	}, f);

	EXPECT_EQ(2, i);
	EXPECT_TRUE(sizeof(f._1) == size_of[0] and sizeof(f._2) == size_of[1]);
	EXPECT_TRUE(alignof(short) == align_of[0] and alignof(int) == align_of[1]);
	EXPECT_TRUE(0 == offset_of[0] and 4 == offset_of[1]);
	EXPECT_TRUE(std::string_view{"_1"} == name[0] and std::string_view{"_2"} == name[1]);
	EXPECT_TRUE(f._1 == value[0] and f._2 == value[1]);
}

/// the (de)serialization of the scheduler performance data (`thread/performance.h`)
TEST(simple, serialize) {
	using namespace cryptanalysislib;
	SchedulerPerformance p;
	p.number_active_threads = 3;
	p.number_enqueud_tasks = -7;
	p.schedulerThreadLoad.resize(2);
	p.schedulerThreadLoad[0].gather();
	p.schedulerThreadLoad[1].tid = 123456789012345ull;
	p.schedulerThreadLoad[1].data.ru_utime.tv_sec = 17;
	p.schedulerThreadLoad[1].data.ru_nivcsw = 99;

	std::string data;
	internal::serialize(data, p);
	EXPECT_EQ(data.find('\n'), std::string::npos);

	std::string_view in = data;
	SchedulerPerformance q;
	ASSERT_TRUE(internal::deserialize(in, q));
	EXPECT_EQ(q.number_active_threads, 3);
	EXPECT_EQ(q.number_enqueud_tasks, -7);
	ASSERT_EQ(q.schedulerThreadLoad.size(), 2u);
	for (size_t i = 0; i < 2; i++) {
		EXPECT_EQ(q.schedulerThreadLoad[i].tid, p.schedulerThreadLoad[i].tid);
		EXPECT_EQ(q.schedulerThreadLoad[i].usertime(), p.schedulerThreadLoad[i].usertime());
		EXPECT_EQ(q.schedulerThreadLoad[i].systime(), p.schedulerThreadLoad[i].systime());
		EXPECT_EQ(q.schedulerThreadLoad[i].maxrss(), p.schedulerThreadLoad[i].maxrss());
		EXPECT_EQ(q.schedulerThreadLoad[i].nivcsw(), p.schedulerThreadLoad[i].nivcsw());
	}

	// truncated input
	std::string_view truncated = std::string_view{data}.substr(0, data.size() / 2);
	EXPECT_FALSE(internal::deserialize(truncated, q));
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
