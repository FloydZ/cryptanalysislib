#ifndef CRYPTANALYSISLIB_REFLECTION_INTERNAL_H
#define CRYPTANALYSISLIB_REFLECTION_INTERNAL_H

#include "string/stringliteral.h"
using namespace cryptanalysislib::internal;

namespace cryptanalysislib::reflection {
	/// To be used inside visitor patterns
	template <class>
	inline constexpr bool always_false_v = false;


	/// Helper class that can be passed to a field
	/// to trigger the default value of the type.
	struct Default {};
	inline static const auto default_value = Default{};

	///
	/// \tparam T
	template<class T>
	struct StdArrayType {
		using Type = T;
	};

	template<class T, size_t _n>
	struct StdArrayType<T[_n]> {
		using Type = std::array<typename StdArrayType<std::remove_cvref_t<T>>::Type, _n>;
		using ValueType = std::remove_cvref_t<T>;
		constexpr static size_t size = _n;
	};

	template<class T>
	using to_std_array_t = StdArrayType<T>::Type;

	template<class T>
	constexpr auto to_std_array(T &&_t) noexcept {
		using Type = std::remove_cvref_t<T>;
		if constexpr (std::is_array_v<Type>) {
			constexpr size_t n = StdArrayType<Type>::size;
			const auto fct = [&]<std::size_t... _i>(std::index_sequence<_i...>) {
				return to_std_array_t<Type>({to_std_array(
				        std::forward<typename StdArrayType<Type>::ValueType>(_t[_i]))...});
			};
			return fct(std::make_index_sequence<n>());
		} else {
			return std::forward<T>(_t);
		}
	}

	template<class T>
	constexpr auto to_std_array(const T &_t) noexcept {
		using Type = std::remove_cvref_t<T>;
		if constexpr (std::is_array_v<Type>) {
			constexpr size_t n = StdArrayType<Type>::size;
			const auto fct = [&]<std::size_t... _i>(std::index_sequence<_i...>) {
				return to_std_array_t<Type>({to_std_array(_t[_i])...});
			};
			return fct(std::make_index_sequence<n>());
		} else {
			return _t;
		}
	}


	template<class T>
	    requires std::is_array_v<T>
	struct Array {
		using Type = T;
		using StdArrayType = to_std_array_t<T>;

		constexpr Array() = default;
		constexpr Array(const StdArrayType &_arr) noexcept : arr_(_arr) {}
		constexpr Array(StdArrayType &&_arr) : arr_(std::move(_arr)) {}
		constexpr Array(const T &_arr) : arr_(to_std_array(_arr)) {}
		constexpr Array(T &&_arr) : arr_(to_std_array(_arr)) {}

		~Array() = default;

		StdArrayType arr_;
	};

	/// helper
	template<class T>
	struct wrap_in_rfl_array {
		using type = T;
	};

	template<class T>
	    requires std::is_array_v<T>
	struct wrap_in_rfl_array<T> {
		using type = Array<T>;
	};

	template<class T>
	using wrap_in_rfl_array_t = typename wrap_in_rfl_array<T>::type;

	///
	/// \tparam _field_name
	/// \tparam Fields
	/// \tparam I
	/// \return
	template<StringLiteral _field_name, class Fields, int I = 0>
	constexpr static int find_index() noexcept {
		using FieldType =
		        std::remove_cvref_t<typename std::tuple_element<I, Fields>::type>;

		constexpr bool name_i_matches = (FieldType::name_ == _field_name);

		if constexpr (name_i_matches) {
			return I;
		} else {
			constexpr bool out_of_range = I + 1 == std::tuple_size_v<Fields>;
			static_assert(!out_of_range, "Field name not found!");
			if constexpr (out_of_range) {
				// This is to avoid very confusing error messages.
				return I;
			} else {
				return find_index<_field_name, Fields, I + 1>();
			}
		}
	}

	///
	/// \tparam Fields
	/// \tparam _i
	/// \tparam _j
	/// \return
	template<class Fields, int _i = 1, int _j = 0>
	constexpr inline bool no_duplicate_field_names_() {
		constexpr auto num_fields = std::tuple_size_v<Fields>;

		if constexpr (num_fields <= 1) {
			return true;
		} else {
			if constexpr (_i == num_fields) {
				return true;
			} else if constexpr (_j == -1) {
				return no_duplicate_field_names_<Fields, _i + 1, _i>();
			} else {
				using FieldType1 =
				        std::remove_cvref_t<typename std::tuple_element<_i, Fields>::type>;
				using FieldType2 =
				        std::remove_cvref_t<typename std::tuple_element<_j, Fields>::type>;

				constexpr auto field_name_i = FieldType1::name_;
				constexpr auto field_name_j = FieldType2::name_;

				constexpr bool no_duplicate = (field_name_i != field_name_j);

				static_assert(no_duplicate, "Duplicate field names are not allowed");

				return no_duplicate && no_duplicate_field_names_<Fields, _i, _j - 1>();
			}
		}
	}



	/*
We infer the number of fields using by figuring out how many fields
we need to construct it. This is done by implementing the constructible
concept, see below.

	However, there is a problem with C arrays. Suppose you have a struct
	         like this:

	    struct A{
		int arr[3];
	};

	Then, the struct can be initialized like this:

	    const auto a = A{1, 2, 3};

This is a problem, because a naive logic would believe that A
has three fields, when in fact it has only one.

That is why we use the constructible concept to get the maximum
possible number of fields and then try to subdivide them into arrays
in order to figure out which of these fields is in fact an array.

Basically, for every field there is, we try to squeeze as many variables into
the potential array as we can without missing variables in subsequent fields.
This is the purpose of get_nested_array_size().
*/


#if __GNUC__
#ifndef __clang__
#pragma GCC system_header
#endif
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#pragma clang diagnostic ignored "-Wundefined-inline"
#endif

	struct any {
		any(std::size_t);
		template <typename T>
		constexpr operator T() const noexcept;
	};

	template <typename T>
	struct CountFieldsHelper {
		template <std::size_t n>
		static consteval bool constructible() {
			return []<std::size_t... is>(std::index_sequence<is...>) {
				return requires { T{any(is)...}; };
			}
			(std::make_index_sequence<n>());
		}

		template <std::size_t l, std::size_t nested, std::size_t r>
		static consteval bool constructible_with_nested() {
			return []<std::size_t... i, std::size_t... j, std::size_t... k>(
			               std::index_sequence<i...>, std::index_sequence<j...>,
			               std::index_sequence<k...>) {
				return requires { T{any(i)..., {any(j)...}, any(k)...}; };
			}
			(std::make_index_sequence<l>(), std::make_index_sequence<nested>(),
			 std::make_index_sequence<r>());
		}

		template <std::size_t n = 0>
		static consteval std::size_t count_max_fields() {
			static_assert(n <= static_cast<std::size_t>(sizeof(T)));
			if constexpr (constructible<n>() && !constructible<n + 1>()) {
				return n;
			} else {
				return count_max_fields<n + 1>();
			}
		}

		template <std::size_t index, std::size_t size, std::size_t rest>
		static consteval std::size_t get_nested_array_size() {
			if constexpr (size < 1) {
				return 1;
			} else if constexpr (constructible_with_nested<index, size, rest>() &&
			                     !constructible_with_nested<index, size, rest + 1>()) {
				return size;
			} else {
				return get_nested_array_size<index, size - 1, rest + 1>();
			}
		}

		template <std::size_t index, std::size_t max>
		static consteval std::size_t count_fields_impl() {
			static_assert(index <= max);
			if constexpr (index == max) {
				return 0;
			} else {
				return 1 +
				       count_fields_impl<
				               index + get_nested_array_size<index, max - index, 0>(), max>();
			}
		}

		static consteval std::size_t count_fields() {
			return count_fields_impl<0, count_max_fields()>();
		}
	};

	template <class T>
	constexpr std::size_t num_fields = CountFieldsHelper<T>::count_fields();




	template <class T>
	consteval auto get_type_name_str_view() {
		// Unfortunately, we cannot avoid the use of a compiler-specific macro for
		// Clang on Windows. For all other compilers, function_name works as intended.
#if defined(__clang__) && defined(_MSC_VER)
		const auto func_name = std::string_view{__PRETTY_FUNCTION__};
#else
		const auto func_name =
		        std::string_view{std::source_location::current().function_name()};
#endif
#if defined(__clang__)
		const auto split = func_name.substr(0, func_name.size() - 1);
		return split.substr(split.find("T = ") + 4);
#elif defined(__GNUC__)
		const auto split = func_name.substr(0, func_name.size() - 1);
		return split.substr(split.find("T = ") + 4);
#elif defined(_MSC_VER)
		auto split = func_name.substr(0, func_name.size() - 7);
		split = split.substr(split.find("get_type_name_str_view<") + 23);
		auto pos = split.find(" ");
		if (pos != std::string_view::npos) {
			return split.substr(pos + 1);
		}
		return split;
#else
		static_assert(
		        false,
		        "You are using an unsupported compiler. Please use GCC, Clang "
		        "or MSVC or explicitly tag your structs using 'Tag' or 'Name'.");
#endif
	}

	template <class T>
	consteval auto get_type_name() {
		static_assert(get_type_name_str_view<int>() == "int",
		              "Expected 'int', got something else.");
		constexpr auto name = get_type_name_str_view<T>();
		const auto to_str_lit = [&]<auto... Ns>(std::index_sequence<Ns...>) {
			return StringLiteral<sizeof...(Ns) + 1>{name[Ns]...};
		};
		return to_str_lit(std::make_index_sequence<name.size()>{});
	}







		template <std::size_t n>
		struct tuple_view_helper {
			template <class T>
			static constexpr auto tuple_view(T&) {
				static_assert(
						always_false_v<T>,
						"\n\nThis error occurs for one of two reasons:\n\n"
						"1) You have created a struct with more than 100 fields, which is "
						"unsupported. Please split up your struct into several "
						"smaller structs and then use rfl::Flatten<...> to combine them. "
						"Refer "
						"to the documentation on rfl::Flatten<...> for details.\n\n"
						"2) You have added a custom constructor to your struct, which you "
						"shouldn't do either. Please refer to the sections on custom "
						"classes or custom parsers in the documentation "
						"for solutions to this problem.\n\n");
			}
		};

		template <>
		struct tuple_view_helper<0> {
			static constexpr auto tuple_view(auto&) { return std::tie(); }
		};

	#define RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(n, ...) \
	  template <>                                         \
	  struct tuple_view_helper<n> {                       \
		static constexpr auto tuple_view(auto& t) {       \
		  auto& [__VA_ARGS__] = t;                        \
		  return std::tie(__VA_ARGS__);                   \
		}                                                 \
	  }

		/*The following boilerplate code was generated using a Python script:
	macro = "RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER"
	with open("generated_code4.cpp", "w", encoding="utf-8") as codefile:
		codefile.write(
			"\n".join(
				[
					f"{macro}({i}, {', '.join([f'f{j}' for j in range(i)])});"
					for i in range(1, 101)
				]
			)
		)
	*/

		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(1, f0);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(2, f0, f1);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(3, f0, f1, f2);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(4, f0, f1, f2, f3);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(5, f0, f1, f2, f3, f4);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(6, f0, f1, f2, f3, f4, f5);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(7, f0, f1, f2, f3, f4, f5, f6);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(8, f0, f1, f2, f3, f4, f5, f6, f7);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(9, f0, f1, f2, f3, f4, f5, f6, f7, f8);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(10, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(11, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(12, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(13, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(14, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(15, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(16, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(17, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(18, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(19, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(20, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(21, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(22, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(23, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(24, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(25, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(26, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(27, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(28, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(29, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(30, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(31, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(32, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(33, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(34, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(35, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(36, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(37, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(38, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(39, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(40, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(41, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(42, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(43, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(44, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(45, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(46, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(47, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(48, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(49, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(50, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(51, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(52, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(53, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(54, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(55, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(56, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54, f55);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(57, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54, f55, f56);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				58, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				59, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				60, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				61, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(62, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54, f55, f56,
											  f57, f58, f59, f60, f61);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(63, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54, f55, f56,
											  f57, f58, f59, f60, f61, f62);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(64, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54, f55, f56,
											  f57, f58, f59, f60, f61, f62, f63);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(65, f0, f1, f2, f3, f4, f5, f6, f7, f8,
											  f9, f10, f11, f12, f13, f14, f15, f16,
											  f17, f18, f19, f20, f21, f22, f23, f24,
											  f25, f26, f27, f28, f29, f30, f31, f32,
											  f33, f34, f35, f36, f37, f38, f39, f40,
											  f41, f42, f43, f44, f45, f46, f47, f48,
											  f49, f50, f51, f52, f53, f54, f55, f56,
											  f57, f58, f59, f60, f61, f62, f63, f64);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				66, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				67, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				68, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				69, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				70, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				71, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				72, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				73, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				74, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				75, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				76, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				77, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				78, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				79, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				80, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				81, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				82, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				83, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				84, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				85, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				86, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				87, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				88, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				89, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				90, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				91, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				92, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				93, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				94, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				95, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93, f94);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				96, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93, f94, f95);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				97, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93, f94, f95, f96);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				98, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93, f94, f95, f96, f97);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				99, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93, f94, f95, f96, f97, f98);
		RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER(
				100, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
				f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
				f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
				f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
				f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
				f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
				f91, f92, f93, f94, f95, f96, f97, f98, f99);

	#undef RFL_INTERNAL_DEFINE_TUPLE_VIEW_HELPER

		template <class T>
		constexpr auto tuple_view(T& t) {
			return tuple_view_helper<num_fields<T>>::tuple_view(t);
		}

		template <class T, typename F>
		constexpr auto bind_to_tuple(T& _t, const F& _f) {
			auto view = tuple_view(_t);
			return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
				return std::make_tuple(_f(std::get<Is>(view))...);
			}
			(std::make_index_sequence<std::tuple_size_v<decltype(view)>>());
		}









#if __GNUC__
#ifndef __clang__
#pragma GCC system_header
#endif
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#pragma clang diagnostic ignored "-Wundefined-internal"
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 7631)
#endif

	    template <class T>
	    struct wrapper {
		    const T value;
		    static const wrapper<T> report_if_you_see_a_link_error_with_this_object;
	    };

	    template <class T>
	    consteval const T& get_fake_object() noexcept {
		    return wrapper<T>::report_if_you_see_a_link_error_with_this_object.value;
	    }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif


	    template <class T, std::size_t n>
	    struct fake_object_tuple_view_helper {
		    static consteval auto tuple_view() {
			    static_assert(
			            always_false_v<T>,
			            "\n\nThis error occurs for one of two reasons:\n\n"
			            "1) You have created a struct with more than 100 fields, which is "
			            "unsupported. Please split up your struct into several "
			            "smaller structs and then use rfl::Flatten<...> to combine them. "
			            "Refer "
			            "to the documentation on rfl::Flatten<...> for details.\n\n"
			            "2) You have added a custom constructor to your struct, which you "
			            "shouldn't do either. Please refer to the sections on custom "
			            "classes or custom parsers in the documentation "
			            "for solutions to this problem.\n\n");
		    }
	    };

	    template <class T>
	    struct fake_object_tuple_view_helper<T, 0> {
		    static consteval auto tuple_view() { return std::tie(); }
	    };

#define RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(n, ...)            \
  template <class T>                                                         \
  struct fake_object_tuple_view_helper<T, n> {                               \
    static consteval auto tuple_view() {                                     \
      const auto& [__VA_ARGS__] = get_fake_object<std::remove_cvref_t<T>>(); \
      const auto ref_tup = std::tie(__VA_ARGS__);                            \
      const auto get_ptrs = [](const auto&... _refs) {                       \
        return std::make_tuple(&_refs...);                                   \
      };                                                                     \
      return std::apply(get_ptrs, ref_tup);                                  \
    }                                                                        \
  }

	    /*The following boilerplate code was generated using a Python script:
macro = "RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER"
with open("generated_code4.cpp", "w", encoding="utf-8") as codefile:
    codefile.write(
        "\n".join(
            [
                f"{macro}({i}, {', '.join([f'f{j}' for j in range(i)])});"
                for i in range(1, 101)
            ]
        )
    )
*/

	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(1, f0);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(2, f0, f1);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(3, f0, f1, f2);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(4, f0, f1, f2, f3);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(5, f0, f1, f2, f3, f4);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(6, f0, f1, f2, f3, f4, f5);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(7, f0, f1, f2, f3, f4, f5,
	                                                      f6);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(8, f0, f1, f2, f3, f4, f5, f6,
	                                                      f7);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(9, f0, f1, f2, f3, f4, f5, f6,
	                                                      f7, f8);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(10, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(11, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(12, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(13, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11,
	                                                      f12);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(14, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(15, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(16, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(17, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(18, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(19, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(20, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(21, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(22, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(23, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(24, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(25, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(26, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(27, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(28, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(29, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(30, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(31, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(32, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30,
	                                                      f31);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(33, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30,
	                                                      f31, f32);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(34, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30,
	                                                      f31, f32, f33);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(35, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30,
	                                                      f31, f32, f33, f34);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(36, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30,
	                                                      f31, f32, f33, f34, f35);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(37, f0, f1, f2, f3, f4, f5,
	                                                      f6, f7, f8, f9, f10, f11, f12,
	                                                      f13, f14, f15, f16, f17, f18,
	                                                      f19, f20, f21, f22, f23, f24,
	                                                      f25, f26, f27, f28, f29, f30,
	                                                      f31, f32, f33, f34, f35, f36);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            38, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            39, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            40, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            41, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            42, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            43, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            44, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            45, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            46, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            47, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            48, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            49, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            50, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            51, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            52, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            53, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            54, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            55, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            56, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            57, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            58, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            59, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            60, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            61, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            62, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            63, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            64, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            65, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            66, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            67, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            68, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            69, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            70, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            71, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            72, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            73, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            74, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            75, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            76, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            77, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            78, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            79, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            80, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            81, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            82, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            83, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            84, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            85, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            86, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            87, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            88, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            89, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            90, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            91, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            92, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            93, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            94, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            95, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93, f94);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            96, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93, f94, f95);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            97, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93, f94, f95, f96);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            98, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93, f94, f95, f96, f97);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            99, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93, f94, f95, f96, f97, f98);
	    RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER(
	            100, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15,
	            f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
	            f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45,
	            f46, f47, f48, f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60,
	            f61, f62, f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75,
	            f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90,
	            f91, f92, f93, f94, f95, f96, f97, f98, f99);

#undef RFL_INTERNAL_DEFINE_FAKE_OBJECT_TUPLE_VIEW_HELPER

	    template <class T>
	    consteval auto bind_fake_object_to_tuple() {
		    return fake_object_tuple_view_helper<T, num_fields<T>>::tuple_view();
	    }

}
#endif//CRYPTANALYSISLIB_INTERNAL_H
