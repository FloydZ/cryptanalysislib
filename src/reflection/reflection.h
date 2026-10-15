#ifndef CRYTPANALYSISLIB_REFLECTION
#define CRYTPANALYSISLIB_REFLECTION

// NOTE: old compilers have a problem with this, as `std::string_view` is not
// 		fully constexpr.

/// C++20 static reflection of aggregates and enums: number, names, types,
/// offsets of the members, type names and enum names.
/// Source: https://github.com/qlibs/reflect (v1.2.5). Differences:
/// 	- the self tests (run upon include in the original) are in
/// 		`tests/reflection/simple.cpp`
/// 	- `REFLECT_ENUM_MIN` = -1 and `REFLECT_ENUM_MAX` = 1024 by default
///
/// Usage:
/// \code
/// 	struct foo { int a; E b; };
/// 	constexpr auto f = foo{.a = 42, .b = B};
/// 	static_assert(2 == reflect::size(f));
/// 	static_assert("a"sv == reflect::member_name<0>(f));
/// 	static_assert(42 == reflect::get<0>(f));
/// 	static_assert(42 == reflect::get<"a">(f));
/// 	static_assert("B"sv == reflect::enum_name(f.b));
/// 	reflect::for_each([&](auto I) {
/// 		std::cout << reflect::member_name<I>(f) << reflect::get<I>(f);
/// 	}, f);
/// \endcode
///
/// Modifications written by Floyd Zweydinger
///
/// The MIT License (MIT)
///
/// Copyright (c) 2024 Kris Jusiak <kris@jusiak.net>
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.

// NOTE: defines the library feature macros (`__cpp_lib_source_location`)
#include <version>

#if !defined(__cpp_lib_source_location)
// #warning "[warning][reflect] __cpp_lib_source_location not supported! "
#elif defined(__CUDACC__)
// NOTE: not available, as nvcc formats `std::source_location::function_name()`
// 	differently, e.g. `member_name` fails to compile
#elif __cplusplus <= 201709L
#warning "[warning][reflect] concepts not supported! "
#elif !defined(__cpp_rvalue_references)
#warning "[warning][reflect] __cpp_rvalue_references not supported!"
#elif !defined(__cpp_decltype)
#warning "[warning][reflect] __cpp_decltype not supported!"
#elif !defined(__cpp_decltype_auto)
#warning "[warning][reflect] __cpp_decltype_auto not supported!"
#elif !defined(__cpp_return_type_deduction)
#warning "[warning][reflect] __cpp_return_type_deduction not supported!"
#elif !defined(__cpp_deduction_guides)
#warning "[warning][reflect] __cpp_deduction_guides not supported!"
#elif !defined(__cpp_generic_lambdas)
#warning "[warning][reflect] __cpp_generic_lambdas not supported!"
#elif !defined(__cpp_constexpr)
#warning "[warning][reflect] __cpp_constexpr not supported!"
#elif !defined(__cpp_if_constexpr)
#warning "[warning][reflect] __cpp_if_constexpr not supported!"
#elif !defined(__cpp_alias_templates)
#warning "[warning][reflect] __cpp_alias_templates not supported!"
#elif !defined(__cpp_variadic_templates)
#warning "[warning][reflect] __cpp_variadic_templates not supported!"
#elif !defined(__cpp_fold_expressions)
#warning "[warning][reflect] __cpp_fold_expressions not supported!"
#elif !defined(__cpp_static_assert)
#warning "[warning][reflect] __cpp_static_assert not supported!"
#elif !defined(__cpp_concepts)
#warning "[warning][reflect] __cpp_concepts not supported!"
#elif !defined(__cpp_nontype_template_args)
#warning "[warning][reflect] __cpp_nontype_template_args not supported!"
#elif !defined(__cpp_nontype_template_parameter_auto)
#warning "[warning][reflect] __cpp_nontype_template_parameter_auto not supported!"
#elif !__has_include(<array>)
#warning "[warning][reflect] <array> not found!"
#elif !__has_include(<string_view>)
#warning "[warning][reflect] <string_view> not found!"
#elif !__has_include(<source_location>)
#warning "[warning][reflect] <source_location> not found!"
#elif !__has_include(<type_traits>)
#warning "[warning][reflect] <type_traits> not found!"
#elif !__has_include(<utility>)
#warning "[warning][reflect] <utility> not found!"
#elif !__has_include(<tuple>)
#warning "[warning][reflect] <tuple> not found!"
#else

// it's globally available
#define CRYPTANALYSISLIB_REFLECTIONS_AVAILABLE

#include <array>
#include <cstdint>
#include <limits>
#include <source_location>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

/// smallest value considered by `enum_name` (can be overridden, e.g.
/// `-DREFLECT_ENUM_MIN=0`, or per enum via an `enum_min(E)` overload)
#ifndef REFLECT_ENUM_MIN
#define REFLECT_ENUM_MIN -1
#endif

/// largest value considered by `enum_name` (can be overridden, e.g.
/// `-DREFLECT_ENUM_MAX=32`, or per enum via an `enum_max(E)` overload)
#ifndef REFLECT_ENUM_MAX
#define REFLECT_ENUM_MAX 1024
#endif

namespace {
	template<bool Cond> struct REFLECT_FWD_LIKE { template<class T> using type = std::remove_reference_t<T>&&; };
	template<> struct REFLECT_FWD_LIKE<true> { template<class T> using type = std::remove_reference_t<T>&; };
} // to speed up compilation times

#define REFLECT_FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)
#define REFLECT_FWD_LIKE(T, ...) static_cast<typename ::REFLECT_FWD_LIKE<::std::is_lvalue_reference_v<T>>::template type<decltype(__VA_ARGS__)>>(__VA_ARGS__)

/// NOTE: has to be in the global namespace. Its name is searched in the
/// 	output of `std::source_location::function_name()`.
struct REFLECT_STRUCT { void* MEMBER; enum class ENUM { VALUE }; };

namespace reflect::inline v1_2_5 {
namespace detail {
	template<class T> extern const T ext{};
	struct any { template<class T> operator T() const noexcept; };
	template<class T> struct any_except_base_of { template<class U> requires (not std::is_base_of_v<U, T>) operator U() const noexcept; };
	template<auto> struct auto_ { constexpr explicit(false) auto_(const auto&...) noexcept { } };
	template<class T> struct ref { T& ref_; };
	template<class T> ref(T&) -> ref<T>;

	/// returns the N-th argument (perfectly forwarded)
	template<std::size_t N>
	inline constexpr auto nth_pack_element = []<auto... Ns>(std::index_sequence<Ns...>) noexcept -> decltype(auto) {
		return [](const auto_<Ns>&..., auto&& nth, const auto&...) noexcept -> decltype(auto) { return REFLECT_FWD(nth); };
	}(std::make_index_sequence<N>{});

	template<auto...  Vs> [[nodiscard]] constexpr auto function_name() noexcept -> std::string_view { return std::source_location::current().function_name(); }
	template<class... Ts> [[nodiscard]] constexpr auto function_name() noexcept -> std::string_view { return std::source_location::current().function_name(); }

	/// the compiler specific prefix/suffix of `function_name` around a type
	template<class T>
	struct type_name_info {
		static constexpr auto name = function_name<int>();
		static constexpr auto begin = name.find("int");
		static constexpr auto end = name.substr(begin+std::size(std::string_view{"int"}));
	};

	template<class T> requires std::is_class_v<T>
	struct type_name_info<T> {
		static constexpr auto name = function_name<REFLECT_STRUCT>();
		static constexpr auto begin = name.find("REFLECT_STRUCT");
		static constexpr auto end = name.substr(begin+std::size(std::string_view{"REFLECT_STRUCT"}));
	};

	template<class T> requires std::is_enum_v<T>
	struct type_name_info<T> {
		static constexpr auto name = function_name<REFLECT_STRUCT::ENUM>();
		static constexpr auto begin = name.find("REFLECT_STRUCT::ENUM");
		static constexpr auto end = name.substr(begin+std::size(std::string_view{"REFLECT_STRUCT::ENUM"}));
	};

	/// the compiler specific prefix/suffix of `function_name` around an enumerator
	struct enum_name_info {
		static constexpr auto name = function_name<REFLECT_STRUCT::ENUM::VALUE>();
		static constexpr auto begin = name.find("REFLECT_STRUCT::ENUM::VALUE");
		static constexpr auto end = std::size(name)-(name.find("REFLECT_STRUCT::ENUM::VALUE")+std::size(std::string_view{"REFLECT_STRUCT::ENUM::VALUE"}));
	};

	/// the compiler specific prefix/suffix of `function_name` around a member
	struct member_name_info {
		static constexpr auto name = function_name<ref{ext<REFLECT_STRUCT>.MEMBER}>();
		static constexpr auto begin = name[name.find("MEMBER")-1];
		static constexpr auto end = name.substr(name.find("MEMBER")+std::size(std::string_view{"MEMBER"}));
	};
} // namespace detail

/// string usable as non-type template parameter, e.g. `get<"a">(f)`
template<class T, std::size_t Size>
struct fixed_string {
	constexpr explicit(false) fixed_string(const T* str) {
		for (decltype(Size) i{}; i < Size; ++i) { data[i] = str[i]; }
		data[Size] = T();
	}
	[[nodiscard]] constexpr auto operator<=>(const fixed_string&) const = default;
	[[nodiscard]] constexpr explicit(false) operator std::string_view() const { return {std::data(data), Size}; }
	[[nodiscard]] constexpr auto size() const { return Size; }
	T data[Size + 1u];
};
template<class T, std::size_t Capacity, std::size_t Size = Capacity-1> fixed_string(const T (&str)[Capacity]) -> fixed_string<T, Size>;

/// true for `std::pair` and `std::tuple`
template<typename T>
inline constexpr bool is_unnamed_record_type_v = false;
template<typename...Ts>
inline constexpr bool is_unnamed_record_type_v<std::pair<Ts...>> = true;
template<typename...Ts>
inline constexpr bool is_unnamed_record_type_v<std::tuple<Ts...>> = true;

/// true for aggregates, `std::pair` and `std::tuple`
template<typename T>
inline constexpr bool is_record_like_type_v = std::is_aggregate_v<T> or is_unnamed_record_type_v<T>;

namespace detail {
	/// \return the number of members of `T`, without the (empty) bases
	template<class T, std::size_t Bases = 0u, class... Ts> requires std::is_aggregate_v<T>
	[[nodiscard]] constexpr auto size() -> std::size_t {
		if constexpr (requires { T{Ts{}...}; } and not requires { T{Ts{}..., detail::any{}}; }) {
			return sizeof...(Ts) - Bases;
		} else if constexpr (Bases == sizeof...(Ts) and requires { T{Ts{}...}; } and not requires { T{Ts{}..., detail::any_except_base_of<T>{}}; }) {
			return size<T, Bases + 1u, Ts..., detail::any>();
		} else {
			return size<T, Bases, Ts..., detail::any>();
		}
	}
} // namespace detail

/// \return the number of members of `T`
template<class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto size() noexcept -> std::size_t {
	return detail::size<std::remove_cvref_t<T>>();
}

template<class T> requires is_unnamed_record_type_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto size() noexcept -> std::size_t {
	return std::tuple_size_v<std::remove_cvref_t<T>>;
}

template<class T> requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto size(const T&) noexcept -> std::size_t {
	return detail::size<T>();
}

template<class T> requires is_unnamed_record_type_v<T>
[[nodiscard]] constexpr auto size(const T&) noexcept -> std::size_t {
	return std::tuple_size_v<T>;
}

namespace detail {
	// one overload per number of members (at most 64)
	#define REFLECT_VISIT(N) template<class Fn, class T> [[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t, std::integral_constant<std::size_t, N>)
	#define REFLECT_V(x) REFLECT_FWD_LIKE(T, x)
	template<class Fn, class T> [[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&&, std::integral_constant<std::size_t, 0>) { return REFLECT_FWD(fn)(); }
	REFLECT_VISIT(1) { auto&& [_1] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1)); }
	REFLECT_VISIT(2) { auto&& [_1, _2] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2)); }
	REFLECT_VISIT(3) { auto&& [_1, _2, _3] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3)); }
	REFLECT_VISIT(4) { auto&& [_1, _2, _3, _4] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4)); }
	REFLECT_VISIT(5) { auto&& [_1, _2, _3, _4, _5] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5)); }
	REFLECT_VISIT(6) { auto&& [_1, _2, _3, _4, _5, _6] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6)); }
	REFLECT_VISIT(7) { auto&& [_1, _2, _3, _4, _5, _6, _7] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7)); }
	REFLECT_VISIT(8) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8)); }
	REFLECT_VISIT(9) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9)); }
	REFLECT_VISIT(10) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10)); }
	REFLECT_VISIT(11) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11)); }
	REFLECT_VISIT(12) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12)); }
	REFLECT_VISIT(13) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13)); }
	REFLECT_VISIT(14) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14)); }
	REFLECT_VISIT(15) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15)); }
	REFLECT_VISIT(16) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16)); }
	REFLECT_VISIT(17) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17)); }
	REFLECT_VISIT(18) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18)); }
	REFLECT_VISIT(19) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19)); }
	REFLECT_VISIT(20) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20)); }
	REFLECT_VISIT(21) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21)); }
	REFLECT_VISIT(22) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22)); }
	REFLECT_VISIT(23) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23)); }
	REFLECT_VISIT(24) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24)); }
	REFLECT_VISIT(25) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25)); }
	REFLECT_VISIT(26) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26)); }
	REFLECT_VISIT(27) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27)); }
	REFLECT_VISIT(28) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28)); }
	REFLECT_VISIT(29) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29)); }
	REFLECT_VISIT(30) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30)); }
	REFLECT_VISIT(31) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31)); }
	REFLECT_VISIT(32) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32)); }
	REFLECT_VISIT(33) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33)); }
	REFLECT_VISIT(34) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34)); }
	REFLECT_VISIT(35) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35)); }
	REFLECT_VISIT(36) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36)); }
	REFLECT_VISIT(37) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37)); }
	REFLECT_VISIT(38) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38)); }
	REFLECT_VISIT(39) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39)); }
	REFLECT_VISIT(40) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40)); }
	REFLECT_VISIT(41) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41)); }
	REFLECT_VISIT(42) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42)); }
	REFLECT_VISIT(43) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43)); }
	REFLECT_VISIT(44) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44)); }
	REFLECT_VISIT(45) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45)); }
	REFLECT_VISIT(46) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46)); }
	REFLECT_VISIT(47) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47)); }
	REFLECT_VISIT(48) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48)); }
	REFLECT_VISIT(49) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49)); }
	REFLECT_VISIT(50) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50)); }
	REFLECT_VISIT(51) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51)); }
	REFLECT_VISIT(52) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52)); }
	REFLECT_VISIT(53) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53)); }
	REFLECT_VISIT(54) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54)); }
	REFLECT_VISIT(55) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55)); }
	REFLECT_VISIT(56) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56)); }
	REFLECT_VISIT(57) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57)); }
	REFLECT_VISIT(58) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58)); }
	REFLECT_VISIT(59) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58), REFLECT_V(_59)); }
	REFLECT_VISIT(60) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58), REFLECT_V(_59), REFLECT_V(_60)); }
	REFLECT_VISIT(61) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58), REFLECT_V(_59), REFLECT_V(_60), REFLECT_V(_61)); }
	REFLECT_VISIT(62) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58), REFLECT_V(_59), REFLECT_V(_60), REFLECT_V(_61), REFLECT_V(_62)); }
	REFLECT_VISIT(63) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58), REFLECT_V(_59), REFLECT_V(_60), REFLECT_V(_61), REFLECT_V(_62), REFLECT_V(_63)); }
	REFLECT_VISIT(64) { auto&& [_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64] = REFLECT_FWD(t); return REFLECT_FWD(fn)(REFLECT_V(_1), REFLECT_V(_2), REFLECT_V(_3), REFLECT_V(_4), REFLECT_V(_5), REFLECT_V(_6), REFLECT_V(_7), REFLECT_V(_8), REFLECT_V(_9), REFLECT_V(_10), REFLECT_V(_11), REFLECT_V(_12), REFLECT_V(_13), REFLECT_V(_14), REFLECT_V(_15), REFLECT_V(_16), REFLECT_V(_17), REFLECT_V(_18), REFLECT_V(_19), REFLECT_V(_20), REFLECT_V(_21), REFLECT_V(_22), REFLECT_V(_23), REFLECT_V(_24), REFLECT_V(_25), REFLECT_V(_26), REFLECT_V(_27), REFLECT_V(_28), REFLECT_V(_29), REFLECT_V(_30), REFLECT_V(_31), REFLECT_V(_32), REFLECT_V(_33), REFLECT_V(_34), REFLECT_V(_35), REFLECT_V(_36), REFLECT_V(_37), REFLECT_V(_38), REFLECT_V(_39), REFLECT_V(_40), REFLECT_V(_41), REFLECT_V(_42), REFLECT_V(_43), REFLECT_V(_44), REFLECT_V(_45), REFLECT_V(_46), REFLECT_V(_47), REFLECT_V(_48), REFLECT_V(_49), REFLECT_V(_50), REFLECT_V(_51), REFLECT_V(_52), REFLECT_V(_53), REFLECT_V(_54), REFLECT_V(_55), REFLECT_V(_56), REFLECT_V(_57), REFLECT_V(_58), REFLECT_V(_59), REFLECT_V(_60), REFLECT_V(_61), REFLECT_V(_62), REFLECT_V(_63), REFLECT_V(_64)); }
	#undef REFLECT_V
	#undef REFLECT_VISIT
} // namespace detail

/// calls `fn` with all members of `t` (perfectly forwarded)
/// \code
/// 	struct foo { int a; int b; };
/// 	static_assert(2 == visit([](auto&&... args) { return sizeof...(args); }, foo{}));
/// \endcode
template<class Fn, class T> requires is_record_like_type_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr decltype(auto) visit(Fn&& fn, T&& t) {
#if (__cpp_structured_bindings >= 202601L)
	auto&& [... ts] = REFLECT_FWD(t);
	return REFLECT_FWD(fn)(REFLECT_FWD_LIKE(T, ts)...);
#else
	return detail::visit(REFLECT_FWD(fn), REFLECT_FWD(t), std::integral_constant<std::size_t, size<std::remove_cvref_t<T>>()>{});
#endif
}

/// \return the unqualified name of `T`, e.g. "foo" for `ns::foo<int>`
template<class T>
[[nodiscard]] constexpr auto type_name() noexcept -> std::string_view {
	using type_name_info = detail::type_name_info<std::remove_pointer_t<std::remove_cvref_t<T>>>;
	constexpr std::string_view function_name = detail::function_name<std::remove_pointer_t<std::remove_cvref_t<T>>>();
	constexpr std::string_view qualified_type_name = function_name.substr(type_name_info::begin, function_name.find(type_name_info::end)-type_name_info::begin);
	constexpr std::string_view tmp_type_name = qualified_type_name.substr(0, qualified_type_name.find_first_of("<", 1));
	constexpr std::string_view type_name = tmp_type_name.substr(tmp_type_name.find_last_of("::")+1);
	static_assert(std::size(type_name) > 0u);
	if (std::is_constant_evaluated()) {
		return type_name;
	} else {
		return [&] {
			static constexpr const auto name = fixed_string<std::remove_cvref_t<decltype(type_name[0])>, std::size(type_name)>{std::data(type_name)};
			return std::string_view{name};
		}();
	}
}

template<class T>
[[nodiscard]] constexpr auto type_name(const T&) noexcept -> std::string_view { return type_name<T>(); }

template<class E>
[[nodiscard]] constexpr auto to_underlying(E e) noexcept {
	return static_cast<std::underlying_type_t<E>>(e);
}

namespace detail {
	template<auto V> consteval const auto& data() { return V.data; }
	template<class T, std::size_t Size>
	struct static_vector {
		constexpr static_vector() = default;
		constexpr auto push_back(const T& value) { values_[size_++] = value; }
		[[nodiscard]] constexpr const auto& operator[](auto i) const { return values_[i]; }
		[[nodiscard]] constexpr auto begin() const { return &values_[0]; }
		[[nodiscard]] constexpr auto end() const { return &values_[0] + size_; }
		[[nodiscard]] constexpr auto size() const { return size_; }
		[[nodiscard]] constexpr auto empty() const { return not size_; }
		[[nodiscard]] constexpr auto capacity() const { return Size; }
		std::array<T, Size> values_{};
		std::size_t size_{};
	};

	/// \return the name of the enumerator `N` of `E`
	template<class E, auto N> requires std::is_enum_v<E>
	[[nodiscard]] constexpr auto enum_name() noexcept {
		constexpr auto fn_name = function_name<static_cast<E>(N)>();
		constexpr auto name = fn_name.substr(enum_name_info::begin, std::size(fn_name)-enum_name_info::end-enum_name_info::begin);
		constexpr auto enum_name = name.substr(name.find_last_of("::")+1);
		static_assert(std::size(enum_name) > 0u);
		return data<fixed_string<std::remove_cvref_t<decltype(enum_name[0])>, std::size(enum_name)>{std::data(enum_name)}>();
	}

	/// all values in [Min, Max] which are enumerators of `E`
#if defined(__clang__)
#if (__clang_major__ > 15) and (__clang_major__ < 19)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wenum-constexpr-conversion"
#endif
	template<class E, auto Min, auto Max> requires (std::is_enum_v<E> and Max > Min)
	inline constexpr auto enum_cases = []<auto... Ns>(std::index_sequence<Ns...>) {
		static_vector<std::underlying_type_t<E>, sizeof...(Ns)> cases{};
		([&] {
			if constexpr (requires { function_name<static_cast<E>(Ns+Min)>(); }) {
				const auto fn_name = function_name<static_cast<E>(Ns+Min)>();
				const auto name = fn_name.substr(enum_name_info::begin, std::size(fn_name)-enum_name_info::end-enum_name_info::begin);
				const auto enum_name = name.substr(name.find_last_of("::")+1);
				if (enum_name.find(")") == std::string_view::npos) {
					cases.push_back(std::underlying_type_t<E>(Ns+Min));
				}
			}
		}(), ...);
		return cases;
	}(std::make_index_sequence<Max-Min+1/*inclusive*/>{});
#if (__clang_major__ > 15) and (__clang_major__ < 19)
	#pragma clang diagnostic pop
#endif
#else
	template<class E, auto Min, auto Max> requires (std::is_enum_v<E> and Max > Min)
	inline constexpr auto enum_cases = []<auto... Ns>(std::index_sequence<Ns...>) {
		const auto names = function_name<static_cast<E>(Ns+Min)...>();
		const auto begin = enum_name_info::begin;
		const auto end = std::size(names) - enum_name_info::end;
		static_vector<std::underlying_type_t<E>, sizeof...(Ns)> cases{};
		std::underlying_type_t<E> index{};
		auto valid = true;
		for (auto i = begin; i < end; ++i) {
			if (names[i] == '(' and names[i+1] != ')') {
				valid = false;
			} else if (names[i] == ' ') {
				valid = true;
			} else if (names[i] == ',' or i == end-1) {
				if (valid) { cases.push_back(index+Min); }
				++index;
				valid = true;
			}
		}
		return cases;
	}(std::make_index_sequence<Max-Min+1/*inclusive*/>{});
#endif
} // namespace detail

/// smallest value considered by `enum_name`, overload it for an enum `E`
/// (found via ADL) to change it
template<class E> requires std::is_enum_v<E> [[nodiscard]] consteval auto enum_min(E) -> int { return REFLECT_ENUM_MIN; }

/// largest value considered by `enum_name`, overload it for an enum `E`
/// (found via ADL) to change it
template<class E, class T = std::underlying_type_t<E>> requires std::is_enum_v<E> [[nodiscard]] consteval auto enum_max(E) -> int {
	return std::numeric_limits<T>::max() < REFLECT_ENUM_MAX ? std::numeric_limits<T>::max() : REFLECT_ENUM_MAX;
}

/// array of (value, name) of all enumerators of `E` in [Min, Max]
template<class E, int Min = enum_min(E{}), int Max = enum_max(E{}), auto enum_cases = detail::enum_cases<E, Min, Max>>
inline constexpr auto enumerators = []<auto... Ns>(std::index_sequence<Ns...>) {
	return std::array{std::pair{enum_cases[Ns], detail::enum_name<E, enum_cases[Ns]>()}...};
}(std::make_index_sequence<enum_cases.size()>{});

/// \return the name of the enumerator `e`, or `unknown`
/// \code
/// 	enum class Enum { foo = 1, bar = 2 };
/// 	static_assert(std::string_view{"foo"} == enum_name(Enum::foo));
/// \endcode
template<class E, fixed_string unknown = "", auto Min = enum_min(E{}), auto Max = enum_max(E{})>
	requires (std::is_enum_v<E> and Max > Min)
[[nodiscard]] constexpr auto enum_name(E e) noexcept -> std::string_view {
	if constexpr (constexpr auto cases = enumerators<E, Min, Max>; std::empty(cases)) {
		return unknown;
	} else {
		const auto switch_case = [&]<std::size_t I = 0u>(auto switch_case, auto value) -> std::string_view {
			if constexpr (I == std::size(cases)) {
				return unknown;
			} else {
				switch (value) {
					default:             return switch_case.template operator()<I + 1u>(switch_case, value);
					case cases[I].first: return cases[I].second;
				}
			}
		};
		return switch_case(switch_case, to_underlying(e));
	}
}

/// unique (per type) compile time value
/// \code
/// 	static_assert(type_id_v<int> == type_id_v<int>);
/// 	static_assert(type_id_v<int> != type_id_v<const int&>);
/// \endcode
template<class...>
inline constexpr auto type_id_v = []{};

/// \return a hash of `type_name<T>()`
template<class T>
[[nodiscard]] constexpr auto type_id() noexcept -> std::size_t {
	std::size_t result{};
	for (const auto c : type_name<T>()) { (result ^= c) <<= 1; }
	return result;
}

template<class T>
[[nodiscard]] constexpr auto type_id(const T&) noexcept -> std::size_t { return type_id<T>(); }

/// \return the name of the `N`-th member of `T`
template<std::size_t N, class T> requires (std::is_aggregate_v<std::remove_cvref_t<T>> and N < size<T>())
[[nodiscard]] constexpr auto member_name() noexcept -> std::string_view {
	constexpr std::string_view function_name = detail::function_name<detail::ref{visit(detail::nth_pack_element<N>, detail::ext<std::remove_cvref_t<T>>)}>();
	constexpr std::string_view tmp_member_name = function_name.substr(0, function_name.find(detail::member_name_info::end));
	constexpr std::string_view member_name = tmp_member_name.substr(tmp_member_name.find_last_of(detail::member_name_info::begin)+1);
	static_assert(std::size(member_name) > 0u);
	if (std::is_constant_evaluated()) {
		return member_name;
	} else {
		return [&] {
			static constexpr const auto name = fixed_string<std::remove_cvref_t<decltype(member_name[0])>, std::size(member_name)>{std::data(member_name)};
			return std::string_view{name};
		}();
	}
}

template<std::size_t N, class T> requires (std::is_aggregate_v<T> and N < size<T>())
[[nodiscard]] constexpr auto member_name(const T&) noexcept -> std::string_view {
	return member_name<N, T>();
}

/// \return the `N`-th member of `t` (perfectly forwarded)
template<std::size_t N, class T> requires (is_record_like_type_v<std::remove_cvref_t<T>> and N < size<std::remove_cvref_t<T>>())
[[nodiscard]] constexpr decltype(auto) get(T&& t) noexcept {
	return visit(detail::nth_pack_element<N>, REFLECT_FWD(t));
}

/// type of the `N`-th member of `T`
template<std::size_t N, class T> requires (is_record_like_type_v<T> and N < size<T>())
using member_type = std::remove_cvref_t<decltype(get<N>(std::declval<T>()))>;

namespace detail {
	template<class T, auto Name>
	inline constexpr bool has_member_name_impl = []<auto... Ns>(std::index_sequence<Ns...>) {
		return ((Name == member_name<Ns, T>()) or ...);
	}(std::make_index_sequence<size<T>()>{});

	/// \return the error message for a missing member `Name` of `T`, which
	/// 	suggests the closest (Levenshtein distance) member name
	template<class T, fixed_string Name>
	[[nodiscard]] consteval auto diagnose_member_name() {
		constexpr std::string_view prefix = "`";
		constexpr std::string_view type = type_name<T>();
		if constexpr (size<T>()) {
			constexpr std::string_view infix1 = "` has no data member named `";
			constexpr std::string_view incorrect = Name;
			constexpr std::string_view infix2 = "`. Did you mean `";
			constexpr std::string_view correct = [] {
				constexpr auto distance = [](std::string_view correct) {
					constexpr std::string_view incorrect = Name;
					std::array<std::size_t, incorrect.size() + 1> prev;
					std::array<std::size_t, incorrect.size() + 1> curr{};
					for (decltype(prev.size()) i{}; i < prev.size(); ++i) prev[i] = i;
					for (decltype(correct.size()) i{}; i < correct.size(); ++i) {
						curr[0] = i + 1;
						for (decltype(incorrect.size()) j{}; j < incorrect.size(); ++j) {
							const auto del = prev[j + 1] + 1;
							const auto ins = curr[j] + 1;
							const auto sub = prev[j] + (correct[i] != incorrect[j]);
							const auto min_del_ins = del < ins ? del : ins;
							curr[j + 1] = min_del_ins < sub ? min_del_ins : sub;
						}
						auto temp = curr;
						curr = prev;
						prev = temp;
					}
					return prev.back();
				};
				const auto member_names = []<auto... Ns>(std::index_sequence<Ns...>) {
					return std::array<std::string_view, sizeof...(Ns)>{member_name<Ns, T>()...};
				}(std::make_index_sequence<size<T>()>{});
				auto closest_name = member_names[0];
				auto min_distance = distance(closest_name);
				for (decltype(member_names.size()) n{1}; n < member_names.size(); ++n) {
					const auto nth_member_name = member_names[n];
					const auto nth_distance = distance(nth_member_name);
					if (nth_distance < min_distance) {
						closest_name = nth_member_name;
						min_distance = nth_distance;
					}
				}
				return closest_name;
			}();
			constexpr std::string_view suffix = "`?";
			char message[prefix.size() + type.size() + infix1.size() + incorrect.size() + infix2.size() + correct.size() + suffix.size() + 1];
			auto out = message;
			for (const auto c : prefix) *out++ = c;
			for (const auto c : type) *out++ = c;
			for (const auto c : infix1) *out++ = c;
			for (const auto c : incorrect) *out++ = c;
			for (const auto c : infix2) *out++ = c;
			for (const auto c : correct) *out++ = c;
			for (const auto c : suffix) *out++ = c;
			*out = '\0';
			return fixed_string{message};
		} else {
			constexpr std::string_view suffix = "` has no data members.";
			char message[prefix.size() + type.size() + suffix.size() + 1];
			auto out = message;
			for (const auto c : prefix) *out++ = c;
			for (const auto c : type) *out++ = c;
			for (const auto c : suffix) *out++ = c;
			*out = '\0';
			return fixed_string{message};
		}
	}

	template<auto Message>
	struct print { static constexpr auto value = false; };
	template<class TMessage>
	concept diagnosis = TMessage::value;
	template<class T, auto Name>
	concept member_name_diagnosis = diagnosis<print<diagnose_member_name<T, Name>()>>;
} // namespace detail

/// true if `T` has a member named `Name`
template<class T, fixed_string Name>
concept has_member_name = detail::has_member_name_impl<std::remove_cvref_t<T>, Name> or detail::member_name_diagnosis<std::remove_cvref_t<T>, Name>;

/// \return the index of the member `Name` of `T`
template<fixed_string Name, class T>
	requires (std::is_aggregate_v<std::remove_cvref_t<T>> and has_member_name<std::remove_cvref_t<T>, Name>)
[[nodiscard]] constexpr auto index_of() noexcept -> std::size_t {
	return []<auto... Ns>(std::index_sequence<Ns...>){
		return (((Name == member_name<Ns, std::remove_cvref_t<T>>()) ? Ns : decltype(Ns){}) + ...);
	}(std::make_index_sequence<size<std::remove_cvref_t<T>>()>{});
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<T> and has_member_name<T, Name>)
[[nodiscard]] constexpr auto index_of(const T&) noexcept -> std::size_t {
	return index_of<Name, T>();
}

/// \return the member `Name` of `t` (perfectly forwarded)
template<fixed_string Name, class T>
	requires (std::is_aggregate_v<std::remove_cvref_t<T>> and has_member_name<std::remove_cvref_t<T>, Name>)
[[nodiscard]] constexpr decltype(auto) get(T&& t) noexcept {
	return visit(detail::nth_pack_element<index_of<Name, std::remove_cvref_t<T>>()>, REFLECT_FWD(t));
}

/// \return `R<...>` of all members of `t`, e.g. `to<std::tuple>(t)`. For an
/// 	lvalue `t` the result holds references to the members.
template<template<class...> class R, class T>
	requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto to(T&& t) {
	if constexpr (std::is_lvalue_reference_v<decltype(t)>) {
		return visit([](auto&&... args) { return R<decltype(REFLECT_FWD(args))...>{REFLECT_FWD(args)...}; }, t);
	} else {
		return visit([](auto&&... args) { return R{REFLECT_FWD(args)...}; }, t);
	}
}

/// copies the members with the same name from `src` to `dst`. If `Members`
/// is not empty, only these members are copied.
template<fixed_string... Members, class TSrc, class TDst>
	requires (std::is_aggregate_v<TSrc> and std::is_aggregate_v<TDst>)
constexpr auto copy(const TSrc& src, TDst& dst) -> void {
	constexpr auto contains = []([[maybe_unused]] auto name) {
		return sizeof...(Members) == 0u or ((name == Members) or ...);
	};
	auto dst_view = to<std::tuple>(dst);
	[&]<auto... Ns>(std::index_sequence<Ns...>) {
		([&] {
			if constexpr (contains(member_name<Ns, TDst>()) and requires { std::get<Ns>(dst_view) = get<fixed_string<std::remove_cvref_t<decltype(member_name<Ns, TDst>()[0])>, std::size(member_name<Ns, TDst>())>(std::data(member_name<Ns, TDst>()))>(src); }) {
				std::get<Ns>(dst_view) = get<fixed_string<std::remove_cvref_t<decltype(member_name<Ns, TDst>()[0])>, std::size(member_name<Ns, TDst>())>(std::data(member_name<Ns, TDst>()))>(src);
			}
		}(), ...);
	}(std::make_index_sequence<size<TDst>()>{});
}

/// \return a `R`, whose members are copied (by name) from `t`
template<class R, class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto to(T&& t) {
	R r{};
	copy(REFLECT_FWD(t), r);
	return r;
}

/// \return `sizeof` of the `N`-th member of `T`
template<std::size_t N, class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto size_of() noexcept -> std::size_t {
	return sizeof(std::remove_cvref_t<decltype(get<N>(detail::ext<T>))>);
}

template<std::size_t N, class T> requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto size_of(const T&) noexcept -> std::size_t {
	return size_of<N, T>();
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<std::remove_cvref_t<T>> and has_member_name<std::remove_cvref_t<T>, Name>)
[[nodiscard]] constexpr auto size_of() noexcept -> std::size_t {
	return size_of<index_of<Name, std::remove_cvref_t<T>>(), std::remove_cvref_t<T>>();
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<T> and has_member_name<T, Name>)
[[nodiscard]] constexpr auto size_of(const T&) noexcept -> std::size_t {
	return size_of<index_of<Name, T>(), T>();
}

/// \return `alignof` of the `N`-th member of `T`
template<std::size_t N, class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto align_of() noexcept -> std::size_t {
	return alignof(std::remove_cvref_t<decltype(get<N>(detail::ext<T>))>);
}

template<std::size_t N, class T> requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto align_of(const T&) noexcept -> std::size_t {
	return align_of<N, T>();
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<std::remove_cvref_t<T>> and has_member_name<std::remove_cvref_t<T>, Name>)
[[nodiscard]] constexpr auto align_of() noexcept -> std::size_t {
	return align_of<index_of<Name, std::remove_cvref_t<T>>(), std::remove_cvref_t<T>>();
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<T> and has_member_name<T, Name>)
[[nodiscard]] constexpr auto align_of(const T&) noexcept -> std::size_t {
	return align_of<index_of<Name, T>(), T>();
}

namespace detail {
	template<class T>
	union uninitialized {
		char bytes[sizeof(T)];
		T value;
	};
} // namespace detail

/// \return `offsetof` of the `N`-th member of `T`
template<std::size_t N, class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
[[nodiscard]] constexpr auto offset_of() noexcept -> std::size_t {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#endif
	const auto& unknown = detail::ext<detail::uninitialized<T>>;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
	const void* address = __builtin_addressof(get<N>(unknown.value));
	for (std::size_t i = 0; i < sizeof(unknown.bytes); ++i) {
		if (address == &unknown.bytes[i]) {
			return i;
		}
	}
#if defined(__clang__)
	__builtin_unreachable();
#endif
}

template<std::size_t N, class T> requires std::is_aggregate_v<T>
[[nodiscard]] constexpr auto offset_of(const T&) noexcept -> std::size_t {
	return offset_of<N, T>();
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<std::remove_cvref_t<T>> and has_member_name<std::remove_cvref_t<T>, Name>)
[[nodiscard]] constexpr auto offset_of() noexcept -> std::size_t {
	return offset_of<index_of<Name, std::remove_cvref_t<T>>(), std::remove_cvref_t<T>>();
}

template<fixed_string Name, class T>
	requires (std::is_aggregate_v<T> and has_member_name<T, Name>)
[[nodiscard]] constexpr auto offset_of(const T&) noexcept -> std::size_t {
	return offset_of<index_of<Name, T>(), T>();
}

/// calls `fn(std::integral_constant<std::size_t, I>{})` for every member
/// index `I` of `T`
template<class T, class Fn> requires std::is_aggregate_v<std::remove_cvref_t<T>>
constexpr auto for_each(Fn&& fn) -> void {
	[&]<auto... Ns>(std::index_sequence<Ns...>) {
		(REFLECT_FWD(fn)(std::integral_constant<decltype(Ns), Ns>{}), ...);
	}(std::make_index_sequence<reflect::size<std::remove_cvref_t<T>>()>{});
}

/// \code
/// 	reflect::for_each([&f](const auto I) {
/// 		std::cout << member_name<I>(f) << "=" << get<I>(f);
/// 	}, f);
/// \endcode
template<class Fn, class T> requires std::is_aggregate_v<std::remove_cvref_t<T>>
constexpr auto for_each(Fn&& fn, T&&) -> void {
	[&]<auto... Ns>(std::index_sequence<Ns...>) {
		(REFLECT_FWD(fn)(std::integral_constant<decltype(Ns), Ns>{}), ...);
	}(std::make_index_sequence<reflect::size<std::remove_cvref_t<T>>()>{});
}
} // namespace reflect

#undef REFLECT_FWD_LIKE
#undef REFLECT_FWD
#endif // checks if all includes and features are available
#endif
