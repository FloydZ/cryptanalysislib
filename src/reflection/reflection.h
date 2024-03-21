#ifndef CRYPTANALYSISLIB_REFLECTION_H
#define CRYPTANALYSISLIB_REFLECTION_H

// code from https://github.com/getml/reflect-cpp

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <cstdint>
#include <type_traits>
#include <source_location>

#include "string/stringliteral.h"
#include "reflection/internal.h"
#include "reflection/literal.h"
#include "reflection/flatten.h"
#include "reflection/rename.h"
#include "reflection/field.h"
#include "reflection/name_tuple.h"

using namespace cryptanalysislib::internal;

namespace cryptanalysislib::reflection {
	/// Convenience constructor that doesn't require you
	/// to explitly define the field types.
	template <class... FieldTypes>
	constexpr inline auto make_named_tuple(FieldTypes&&... _args) noexcept {
		return NamedTuple<std::remove_cvref_t<FieldTypes>...>(
		        std::forward<FieldTypes>(_args)...);
	}

	/// Convenience constructor that doesn't require you
	/// to explitly define the field types.
	template <class... FieldTypes>
	constexpr inline auto make_named_tuple(const FieldTypes&... _args) noexcept {
		return NamedTuple<FieldTypes...>(_args...);
	}

	/// Explicit overload for creating empty named tuples.
	inline auto make_named_tuple() { return NamedTuple<>(); }

	namespace internal {
		template <class T>
		constexpr auto to_ptr_tuple(T& _t) {
			if constexpr (std::is_pointer_v<std::remove_cvref_t<T>>) {
				return to_ptr_tuple(*_t);
			} else {
				return bind_to_tuple(_t, [](auto&& x) {
					return std::addressof(std::forward<decltype(x)>(x));
				});
			}
		}

		template <class T>
		using ptr_tuple_t = decltype(to_ptr_tuple(std::declval<T&>()));

		/////
		template <class T>
		class is_rename;

		template <class T>
		class is_rename : public std::false_type {};

		template <StringLiteral _name, class Type>
		class is_rename<Rename<_name, Type>> : public std::true_type {};

		template <class T>
		constexpr bool is_rename_v = is_rename<std::remove_cvref_t<std::remove_pointer_t<T>>>::value;

		/// --------------

		template <class T>
		class is_flatten_field;

		template <class T>
		class is_flatten_field : public std::false_type {};

		template <class T>
		class is_flatten_field<Flatten<T>> : public std::true_type {};

		template <class T>
		constexpr bool is_flatten_field_v = is_flatten_field<std::remove_cvref_t<std::remove_pointer_t<T>>>::value;

		/// ----------------------
		template <class T>
		class is_field;

		template <class T>
		class is_field : public std::false_type {};

		template <StringLiteral _name, class Type>
		class is_field<Field<_name, Type>> : public std::true_type {};

		template <class T>
		constexpr bool is_field_v =
		        is_field<std::remove_cvref_t<std::remove_pointer_t<T>>>::value;

		/// --------------------------------

		template <class T>
		class is_named_tuple;

		template <class T>
		class is_named_tuple : public std::false_type {};

		template <class... Fields>
		class is_named_tuple<NamedTuple<Fields...>> : public std::true_type {};

		template <class T>
		constexpr bool is_named_tuple_v =
				is_named_tuple<std::remove_cvref_t<std::remove_pointer_t<T>>>::value;

		///  -----------

		template <class T>
		struct lit_name;

		template <auto _name>
		struct lit_name<Literal<_name>> {
			constexpr static auto name_ = _name;
		};

		template <class LiteralType>
		constexpr auto lit_name_v = lit_name<LiteralType>::name_;

		/// ====================--------

		/// Returns a rfl::Literal containing the type name of T.
		template <class T>
		using type_name_t = Literal<get_type_name<T>()>;

		//--------


		template <StringLiteral _name, class T>
		constexpr inline auto make_field(T&& _value) noexcept {
			using T0 = std::remove_cvref_t<T>;
			if constexpr (std::is_array_v<T0>) {
				return Field<_name, T0>(Array<T0>(std::forward<T>(_value)));
			} else {
				return Field<_name, T0>(std::forward<T>(_value));
			}
		}

		///
		/// \tparam TupleType
		/// \tparam _i
		/// \return
		template <class TupleType, int _i = 0>
		constexpr bool all_fields_or_flatten() noexcept {
			if constexpr (_i == std::tuple_size_v<TupleType>) {
				return true;
			} else {
				using T = std::remove_cvref_t<std::tuple_element_t<_i, TupleType>>;
				if constexpr (is_flatten_field_v<T>) {
					return all_fields_or_flatten<
							ptr_tuple_t<typename std::remove_pointer_t<T>::Type>>() &&
						   all_fields_or_flatten<TupleType, _i + 1>();
				} else {
					return is_field_v<T> && all_fields_or_flatten<TupleType, _i + 1>();
				}
			}
		}

		//;
		template <class TupleType, int _i = 0>
		constexpr bool some_fields_or_flatten() noexcept {
			if constexpr (_i == std::tuple_size_v<TupleType>) {
				return false;
			} else {
				using T = std::remove_cvref_t<std::tuple_element_t<_i, TupleType>>;
				if constexpr (is_flatten_field_v<T>) {
					return some_fields_or_flatten<
							ptr_tuple_t<typename std::remove_pointer_t<T>::Type>>() ||
						   some_fields_or_flatten<TupleType, _i + 1>();
				} else {
					return is_field_v<T> || some_fields_or_flatten<TupleType, _i + 1>();
				}
			}
		}

		template <class T>
		constexpr bool has_fields() noexcept {
			if constexpr (is_named_tuple_v<T>) {
				return true;
			} else {
				using TupleType = ptr_tuple_t<T>;
				if constexpr (some_fields_or_flatten<TupleType>()) {
					static_assert(
							all_fields_or_flatten<TupleType>(),
							"If some of your fields are annotated using rfl::Field<...>, "
							"then you must annotate all of your fields. "
							"Also, you cannot combine annotated and "
							"unannotated fields using rfl::Flatten<...>.");
					return true;
				} else {
					return false;
				}
			}
		}



		template <class T>
		struct Wrapper {
			using Type = T;
			T v;
		};

		template <class T>
		Wrapper(T) -> Wrapper<T>;

		// This workaround is necessary for clang.
		template <class T>
		constexpr auto wrap(const T& arg) noexcept {
			return Wrapper{arg};
		}

		template <class T, auto ptr>
		consteval auto get_field_name_str_view() noexcept {
			// Unfortunately, we cannot avoid the use of a compiler-specific macro for
			// Clang on Windows. For all other compilers, function_name works as intended.
			#if defined(__clang__) && defined(_MSC_VER)
			const auto func_name = std::string_view{__PRETTY_FUNCTION__};
			#else
			const auto func_name = std::string_view{ std::source_location::current().function_name() };
			#endif
			#if defined(__clang__)
			const auto split = func_name.substr(0, func_name.size() - 2);
			return split.substr(split.find_last_of(".") + 1);
			#elif defined(__GNUC__)
			const auto split = func_name.substr(0, func_name.size() - 2);
			return split.substr(split.find_last_of(":") + 1);
			#elif defined(_MSC_VER)
			const auto split = func_name.substr(0, func_name.size() - 7);
			return split.substr(split.find("value->") + 7);
			#else
			static_assert(false,
						  "You are using an unsupported compiler. Please use GCC, Clang "
						  "or MSVC or switch to the Field-syntax.");
			#endif
		}

		template <class T, auto ptr>
		consteval auto get_field_name_str_lit() noexcept {
			constexpr auto name = get_field_name_str_view<T, ptr>();
			const auto to_str_lit = [&]<auto... Ns>(std::index_sequence<Ns...>) {
				return StringLiteral<sizeof...(Ns) + 1>{name[Ns]...};
			};
			return to_str_lit(std::make_index_sequence<name.size()>{});
		}

		///
		/// \tparam T
		/// \return
		template <class T>
		constexpr auto get_field_names()noexcept;

		///
		/// \tparam T
		/// \tparam ptr
		/// \return
		template <class T, auto ptr>
		constexpr auto get_field_name() noexcept {
			#if defined(__clang__)
			using Type = std::remove_cvref_t<std::remove_pointer_t<
					typename std::remove_pointer_t<decltype(ptr)>::Type>>;
			#else
			using Type = std::remove_cvref_t<std::remove_pointer_t<decltype(ptr)>>;
			#endif

			if constexpr (is_rename_v<Type>) {
				using Name = typename Type::Name;
				return Name();
			} else if constexpr (is_flatten_field_v<Type>) {
				return get_field_names<std::remove_cvref_t<typename Type::Type>>();
			} else {
				return Literal<get_field_name_str_lit<T, ptr>()>();
			}
		}

		///
		/// \tparam _names1
		/// \tparam _names2
		/// \param _lit1
		/// \param _lit2
		/// \return
		template <StringLiteral... _names1, StringLiteral... _names2>
		constexpr auto concat_two_literals(const Literal<_names1...>& _lit1,
								 const Literal<_names2...>& _lit2) noexcept {
		    (void)_lit1;
			(void)_lit2;
			return Literal<_names1..., _names2...>::template from_value<0>();
		}

		template <class Head, class... Tail>
		constexpr auto concat_literals(const Head& _head, const Tail&... _tail) noexcept {
			if constexpr (sizeof...(_tail) == 0) {
				return _head;
			} else {
				return concat_two_literals(_head, concat_literals(_tail...));
			}
		}

		#ifdef __clang__
		#pragma clang diagnostic push
		#pragma clang diagnostic ignored "-Wundefined-var-template"
		#pragma clang diagnostic ignored "-Wundefined-inline"
		#endif

		template <class T>
		#if __GNUC__
		#ifndef __clang__
			[[gnu::no_sanitize_undefined]]
		#endif
		#endif
		constexpr auto get_field_names() noexcept {
			using Type = std::remove_cvref_t<T>;
			if constexpr (std::is_pointer_v<Type>) {
				return get_field_names<std::remove_pointer_t<T>>();
			} else {
				#if defined(__clang__)
				const auto get = []<std::size_t... Is>(std::index_sequence<Is...>) {
					return concat_literals(
							get_field_name<Type, wrap(std::get<Is>(bind_fake_object_to_tuple<T>()))>()...);
				};
				#else
				const auto get = []<std::size_t... Is>(std::index_sequence<Is...>) {
					return concat_literals(
							get_field_name<Type,
										   std::get<Is>(bind_fake_object_to_tuple<T>())>()...);
				};
				#endif
				return get(std::make_index_sequence<num_fields<T>>());
			}
		}

		#ifdef __clang__
		#pragma clang diagnostic pop
		#endif

		/// Returns a Literal containing the field names of struct T.
		template <class T>
		using field_names_t = typename std::invoke_result<decltype(internal::get_field_names<std::remove_cvref_t<T>>)>::type;

		///
		/// \tparam T
		/// \param _t
		/// \return
		template <class T>
		constexpr auto to_ptr_field_tuple(T& _t) noexcept {
			    if constexpr (std::is_pointer_v<std::remove_cvref_t<T>>) {
				    return to_ptr_field_tuple(*_t);
			    } else if constexpr (is_named_tuple_v<T>) {
				    return nt_to_ptr_named_tuple(_t).fields();
			    } else if constexpr (has_fields<T>()) {
				    return bind_to_tuple(_t, [](auto& x) { return to_ptr_field(x); });
			    } else {
				    using FieldNames = field_names_t<T>;
				    auto tup = bind_to_tuple(_t, [](auto& x) { return to_ptr_field(x); });
				    return wrap_in_fields<FieldNames>(std::move(tup));
			    }
		}


		template <class PtrFieldTuple, class... Args>
		constexpr auto flatten_ptr_field_tuple(PtrFieldTuple& _t, Args&&... _args) noexcept {
			constexpr auto i = sizeof...(Args);
			if constexpr (i == std::tuple_size_v<std::remove_cvref_t<PtrFieldTuple>>) {
				return std::tuple_cat(std::forward<Args>(_args)...);
			} else {
				using T = std::tuple_element_t<i, std::remove_cvref_t<PtrFieldTuple>>;
				if constexpr (internal::is_flatten_field<T>::value) {
					const auto subtuple =
					        internal::to_ptr_field_tuple(*std::get<i>(_t).get());
					return flatten_ptr_field_tuple(_t, std::forward<Args>(_args)...,
					                               flatten_ptr_field_tuple(subtuple));
				} else {
					return flatten_ptr_field_tuple(_t, std::forward<Args>(_args)...,
					                               std::make_tuple(std::get<i>(_t)));
				}
			}
		}

		///
		/// \tparam PtrFieldTuple
		/// \param _ptr_field_tuple
		/// \return
		template <class PtrFieldTuple>
		constexpr auto field_tuple_to_named_tuple(PtrFieldTuple& _ptr_field_tuple) noexcept {
			const auto ft_to_nt = []<class... Fields>(const Fields&... _fields) {
				return make_named_tuple(_fields...);
			};

			if constexpr (!has_flatten_fields<std::remove_cvref_t<PtrFieldTuple>>()) {
				return std::apply(ft_to_nt, std::move(_ptr_field_tuple));
			} else {
				const auto flattened_tuple = flatten_ptr_field_tuple(_ptr_field_tuple);
				return std::apply(ft_to_nt, flattened_tuple);
			}
		}

		///
		/// \tparam TupleType
		/// \tparam _i
		/// \return
		template <class TupleType, int _i = 0>
		constexpr bool has_flatten_fields() noexcept {
			if constexpr (_i == std::tuple_size_v<TupleType>) {
				return false;
			} else {
				using T = std::remove_cvref_t<std::tuple_element_t<_i, TupleType>>;
				return is_flatten_field_v<T> || has_flatten_fields<TupleType, _i + 1>();
			}
		}

		///
		///
		/// \tparam FieldNames
		/// \tparam Fields
		/// \param _flattened_tuple
		/// \param _fields
		/// \return
		template <class FieldNames, class... Fields>
		constexpr auto copy_flattened_tuple_to_named_tuple(const auto& _flattened_tuple,
		                                         Fields&&... _fields) noexcept {
			constexpr auto size =
			        std::tuple_size_v<std::remove_cvref_t<decltype(_flattened_tuple)>>;
			constexpr auto i = sizeof...(_fields);
			if constexpr (i == size) {
				return make_named_tuple(std::move(_fields)...);
			} else {
				const auto name_literal = FieldNames::template name_of<i>();
				auto new_field = make_field<
				        lit_name_v<std::remove_cvref_t<decltype(name_literal)>>>(
				        std::get<i>(_flattened_tuple));
				return copy_flattened_tuple_to_named_tuple<FieldNames>(
				        _flattened_tuple, std::move(_fields)..., std::move(new_field));
			}
		}




		template <class PtrTuple, class... Args>
		constexpr auto flatten_ptr_tuple(PtrTuple&& _t, Args... _args) noexcept {
			constexpr auto i = sizeof...(Args);
			if constexpr (i == 0 && !has_flatten_fields<PtrTuple>()) {
				return std::forward<PtrTuple>(_t);
			} else if constexpr (i == std::tuple_size_v<std::remove_cvref_t<PtrTuple>>) {
				return std::tuple_cat(std::forward<Args>(_args)...);
			} else {
				using T = std::tuple_element_t<i, std::remove_cvref_t<PtrTuple>>;
				if constexpr (is_flatten_field_v<T>) {
					return flatten_ptr_tuple(
							std::forward<PtrTuple>(_t), std::forward<Args>(_args)...,
							flatten_ptr_tuple(to_ptr_tuple(std::get<i>(_t)->get())));
				} else {
					return flatten_ptr_tuple(std::forward<PtrTuple>(_t),
											 std::forward<Args>(_args)...,
											 std::make_tuple(std::get<i>(_t)));
				}
			}
		}

		template <class T>
		constexpr auto to_flattened_ptr_tuple(T&& _t) noexcept {
			return flatten_ptr_tuple(to_ptr_tuple(_t));
		}

		template <class T>
		using flattened_ptr_tuple_t =
			typename std::invoke_result<decltype(to_flattened_ptr_tuple<T>), T>::type;


		/// Generates a named tuple that contains pointers to the original values in
		/// the struct.
		template <class T>
		constexpr auto to_ptr_named_tuple(T&& _t) noexcept {
			if constexpr (has_fields<std::remove_cvref_t<T>>()) {
				if constexpr (std::is_pointer_v<std::remove_cvref_t<T>>) {
					return to_ptr_named_tuple(*_t);
				} else if constexpr (is_named_tuple_v<std::remove_cvref_t<T>>) {
					return nt_to_ptr_named_tuple(_t);
				} else {
					auto ptr_field_tuple = to_ptr_field_tuple(_t);
					return field_tuple_to_named_tuple(ptr_field_tuple);
				}
			} else {
				using FieldNames = field_names_t<T>;
				auto flattened_ptr_tuple = to_flattened_ptr_tuple(_t);
				return copy_flattened_tuple_to_named_tuple<FieldNames>(flattened_ptr_tuple);
			}
		}

		///
		/// \tparam FieldNames
		/// \tparam j
		/// \tparam Fields
		/// \param _tuple
		/// \param _fields
		/// \return
		template <class FieldNames, int j = 0, class... Fields>
		constexpr auto wrap_in_fields(auto&& _tuple, Fields&&... _fields) noexcept {
			constexpr auto size =
					std::tuple_size_v<std::remove_cvref_t<decltype(_tuple)>>;
			constexpr auto i = sizeof...(_fields);
			if constexpr (i == size) {
				return std::make_tuple(std::move(_fields)...);
			} else {
				auto value = std::move(std::get<i>(_tuple));
				using Type = std::remove_cvref_t<std::remove_pointer_t<decltype(value)>>;
				if constexpr (is_flatten_field_v<Type>) {
					// The problem here is that the FieldNames are already flattened, but this
					// is not, so we need to determine how many field names to skip.
					constexpr auto n_skip = std::tuple_size_v<
							std::remove_cvref_t<flattened_ptr_tuple_t<typename Type::Type>>>;
					return wrap_in_fields<FieldNames, j + n_skip>(
							std::move(_tuple), std::move(_fields)..., std::move(value));
				} else {
					const auto name_literal = FieldNames::template name_of<j>();
					auto new_field = make_field<
							lit_name_v<std::remove_cvref_t<decltype(name_literal)>>>(
							std::move(value));
					return wrap_in_fields<FieldNames, j + 1>(
							std::move(_tuple), std::move(_fields)..., std::move(new_field));
				}
			}
		}

		///
		/// \tparam OriginalStruct
		/// \param _t
		/// \return
		template <class OriginalStruct>
		constexpr auto move_to_field_tuple(OriginalStruct&& _t) noexcept {
			using T = std::remove_cvref_t<OriginalStruct>;
			if constexpr (is_named_tuple_v<T>) {
				return _t.fields();
			} else if constexpr (has_fields<T>()) {
				return bind_to_tuple(_t, [](auto& x) { return std::move(x); });
			} else {
				using FieldNames = field_names_t<T>;
				const auto fct = []<class T>(T& _v) {
					using Type = std::remove_cvref_t<T>;
					if constexpr (std::is_array_v<Type>) {
						return Array<Type>(_v);
					} else {
						return std::move(_v);
					}
				};
				auto tup = bind_to_tuple(_t, fct);
				return wrap_in_fields<FieldNames>(std::move(tup));
			}
		}


		/// Creates a struct of type T from a named tuple.
		/// All fields of the struct must be an rfl::Field.
		template <class T, class NamedTupleType>
		constexpr T copy_from_named_tuple(const NamedTupleType& _n) noexcept {
			auto n = _n;
			return move_from_named_tuple(std::move(n));
		}

		///
		/// \tparam T
		/// \param _t
		/// \return
		template <class T>
		constexpr auto copy_to_field_tuple(const T& _t) noexcept {
			auto t = _t;
			return move_to_field_tuple(std::move(t));
		}

		/// Creates a struct of type T from a tuple by copying the underlying
		/// fields.
		/// \tparam T
		/// \tparam TupleType
		/// \param _t
		/// \return
		template <class T, class TupleType>
		constexpr T copy_from_tuple(const TupleType& _t) noexcept {
			auto t = _t;
			return move_from_tuple<T, TupleType>(std::move(t));
		}




		template <class T>
		using ptr_named_tuple_t = typename std::invoke_result<decltype(to_ptr_named_tuple<T>), T>::type;


		// ----------------------------------------------------------------------------
		template <class NamedTupleType>
		struct Getter;

		/// Default case - anything that cannot be explicitly matched.
		template <class NamedTupleType>
		struct Getter {
		public:
			/// Retrieves the indicated value from the tuple.
			template <int _index>
			static inline auto& get(NamedTupleType& _tup) {
				return std::get<_index>(_tup.values());
			}

			/// Gets a field by name.
			template <StringLiteral _field_name>
			static inline auto& get(NamedTupleType& _tup) {
				constexpr auto index =
				        find_index<_field_name, typename NamedTupleType::Fields>();
				return Getter<NamedTupleType>::template get<index>(_tup);
			}

			/// Gets a field by the field type.
			template <class Field>
			static inline auto& get(NamedTupleType& _tup) {
				constexpr auto index =
				        find_index<Field::name_, typename NamedTupleType::Fields>();
				static_assert(
				        std::is_same<typename std::tuple_element<
				                             index, typename NamedTupleType::Fields>::type::Type,
				                     typename Field::Type>(),
				        "If two fields have the same name, "
				        "their type must be the same as "
				        "well.");
				return Getter<NamedTupleType>::template get<index>(_tup);
			}

			/// Retrieves the indicated value from the tuple.
			template <int _index>
			static inline const auto& get_const(const NamedTupleType& _tup) {
				return std::get<_index>(_tup.values());
			}

			/// Gets a field by name.
			template <StringLiteral _field_name>
			static inline const auto& get_const(const NamedTupleType& _tup) {
				constexpr auto index =
				        find_index<_field_name, typename NamedTupleType::Fields>();
				return Getter<NamedTupleType>::template get_const<index>(_tup);
			}

			/// Gets a field by the field type.
			template <class Field>
			static inline const auto& get_const(const NamedTupleType& _tup) {
				constexpr auto index =
				        find_index<Field::name_, typename NamedTupleType::Fields>();
				static_assert(
				        std::is_same<typename std::tuple_element<
				                             index, typename NamedTupleType::Fields>::type::Type,
				                     typename Field::Type>(),
				        "If two fields have the same name, "
				        "their type must be the same as "
				        "well.");
				return Getter<NamedTupleType>::template get_const<index>(_tup);
			}
		};

		// ----------------------------------------------------------------------------

		/// For handling std::variant.
		template <class... NamedTupleTypes>
		struct Getter<std::variant<NamedTupleTypes...>> {
		public:
			/// Retrieves the indicated value from the tuple.
			template <int _index>
			static inline auto& get(std::variant<NamedTupleTypes...>& _tup) {
				const auto apply = [](auto& _t) -> auto& {
					using NamedTupleType = std::remove_cvref_t<decltype(_t)>;
					return Getter<NamedTupleType>::template get<_index>(_t);
				};
				return std::visit(apply, _tup);
			}

			/// Gets a field by name.
			template <StringLiteral _field_name>
			static inline auto& get(std::variant<NamedTupleTypes...>& _tup) {
				const auto apply = [](auto& _t) -> auto& {
					using NamedTupleType = std::remove_cvref_t<decltype(_t)>;
					return Getter<NamedTupleType>::template get<_field_name>(_t);
				};
				return std::visit(apply, _tup);
			}

			/// Gets a field by the field type.
			template <class Field>
			static inline auto& get(std::variant<NamedTupleTypes...>& _tup) {
				const auto apply = [](auto& _t) -> auto& {
					using NamedTupleType = std::remove_cvref_t<decltype(_t)>;
					return Getter<NamedTupleType>::template get<Field>(_t);
				};
				return std::visit(apply, _tup);
			}

			/// Retrieves the indicated value from the tuple.
			template <int _index>
			static inline const auto& get_const(
			        const std::variant<NamedTupleTypes...>& _tup) {
				const auto apply = [](const auto& _tup) -> const auto& {
					using NamedTupleType = std::remove_cvref_t<decltype(_tup)>;
					return Getter<NamedTupleType>::template get_const<_index>(_tup);
				};
				return std::visit(apply, _tup);
			}

			/// Gets a field by name.
			template <StringLiteral _field_name>
			static inline const auto& get_const(
			        const std::variant<NamedTupleTypes...>& _tup) {
				const auto apply = [](const auto& _t) -> const auto& {
					using NamedTupleType = std::remove_cvref_t<decltype(_t)>;
					return Getter<NamedTupleType>::template get_const<_field_name>(_t);
				};
				return std::visit(apply, _tup);
			}

			/// Gets a field by the field type.
			template <class Field>
			static inline const auto& get_const(
			        const std::variant<NamedTupleTypes...>& _tup) {
				const auto apply = [](const auto& _t) -> const auto& {
					using NamedTupleType = std::remove_cvref_t<decltype(_t)>;
					return Getter<NamedTupleType>::template get_const<Field>(_t);
				};
				return std::visit(apply, _tup);
			}
		};

		template <class NamedTupleType, class... AlreadyExtracted>
		auto get_meta_fields(AlreadyExtracted&&... _already_extracted) {
			constexpr size_t i = sizeof...(_already_extracted);
			if constexpr (NamedTupleType::size() == i) {
				return std::array<MetaField, i>{std::move(_already_extracted)...};
			} else {
				using FieldType = std::tuple_element_t<i, typename NamedTupleType::Fields>;
				auto name = typename FieldType::Name().str();
				auto type = type_name_t<typename FieldType::Type>().str();
				return get_meta_fields<NamedTupleType>(
				        std::move(_already_extracted)...,
				        MetaField(std::move(name), std::move(type)));
			}
		}


	}  // namespace internal

	template <class FieldTuple>
	auto move_field_tuple_to_named_tuple(FieldTuple&& _field_tuple) {
		const auto ft_to_nt = []<class... Fields>(Fields&&... _fields) {
			return make_named_tuple(std::move(_fields)...);
		};

		if constexpr (!internal::has_flatten_fields<std::remove_cvref_t<FieldTuple>>()) {
			return std::apply(ft_to_nt, std::move(_field_tuple));
		} else {
			auto flattened_tuple =
			        move_and_flatten_field_tuple(std::move(_field_tuple));
			return std::apply(ft_to_nt, std::move(flattened_tuple));
		}
	}

	/// Helper function to retrieve a name at compile time.
	template <class LiteralType, int _value>
	inline constexpr auto name_of() {
		return LiteralType::template name_of<_value>();
	}

	/// Helper function to retrieve a value at compile time.
	template <class LiteralType, StringLiteral _name>
	inline constexpr auto value_of() {
		return LiteralType::template value_of<_name>();
	}

	/// <=> for other Literals with the same fields.
	template <StringLiteral... fields>
	inline auto operator<=>(const Literal<fields...>& _l1,
	                        const Literal<fields...>& _l2) {
		return _l1.value() <=> _l2.value();
	}

	/// <=> for other Literals with different fields.
	template <StringLiteral... fields1,
	          StringLiteral... fields2>
	inline auto operator<=>(const Literal<fields1...>& _l1,
	                        const Literal<fields2...>& _l2) {
		return _l1.name() <=> _l2.name();
	}

	/// <=> for strings.
	template <StringLiteral... other_fields>
	inline auto operator<=>(const Literal<other_fields...>& _l,
	                        const std::string& _str) {
		return _l <=> _str;
	}





	/// Gets a field by index.
	template <int _index, class NamedTupleType>
	inline auto& get(NamedTupleType& _tup) {
	    return internal::Getter<NamedTupleType>::template get<_index>(_tup);
	}

	/// Gets a field by name.
	template <StringLiteral _field_name, class NamedTupleType>
	inline auto& get(NamedTupleType& _tup) {
	    return internal::Getter<NamedTupleType>::template get<_field_name>(_tup);
	}

	/// Gets a field by the field type.
	template <class Field, class NamedTupleType>
	inline auto& get(NamedTupleType& _tup) {
	    return internal::Getter<NamedTupleType>::template get<Field>(_tup);
	}

	/// Gets a field by index.
	template <int _index, class NamedTupleType>
	inline const auto& get(const NamedTupleType& _tup) {
	    return internal::Getter<NamedTupleType>::template get_const<_index>(_tup);
	}

	/// Gets a field by name.
	template <StringLiteral _field_name, class NamedTupleType>
	inline const auto& get(const NamedTupleType& _tup) {
	    return internal::Getter<NamedTupleType>::template get_const<_field_name>(
	            _tup);
	}

	/// Gets a field by the field type.
	template <class Field, class NamedTupleType>
	inline const auto& get(const NamedTupleType& _tup) {
	    return internal::Getter<NamedTupleType>::template get_const<Field>(_tup);
	}



	// ----------------------------------------------------------------------------

	template <class... FieldTypes>
	inline bool operator==(const NamedTuple<FieldTypes...>& _nt1,
						   const NamedTuple<FieldTypes...>& _nt2) {
		return _nt1.values() == _nt2.values();
	}

	template <class... FieldTypes>
	inline bool operator!=(const NamedTuple<FieldTypes...>& _nt1,
						   const NamedTuple<FieldTypes...>& _nt2) {
		return _nt1.values() != _nt2.values();
	}

	template <StringLiteral _name1, class Type1,
			  StringLiteral _name2, class Type2>
	inline auto operator*(const Field<_name1, Type1>& _f1,
						  const Field<_name2, Type2>& _f2) {
		return NamedTuple(_f1, _f2);
	}

	template <StringLiteral _name, class Type, class... FieldTypes>
	inline auto operator*(const NamedTuple<FieldTypes...>& _tup,
						  const Field<_name, Type>& _f) {
		return _tup.add(_f);
	}

	template <StringLiteral _name, class Type, class... FieldTypes>
	inline auto operator*(const Field<_name, Type>& _f,
						  const NamedTuple<FieldTypes...>& _tup) {
		return NamedTuple(_f).add(_tup);
	}

	template <class... FieldTypes1, class... FieldTypes2>
	inline auto operator*(const NamedTuple<FieldTypes1...>& _tup1,
						  const NamedTuple<FieldTypes2...>& _tup2) {
		return _tup1.add(_tup2);
	}

	template <StringLiteral _name1, class Type1,
			  StringLiteral _name2, class Type2>
	inline auto operator*(Field<_name1, Type1>&& _f1,
						  Field<_name2, Type2>&& _f2) {
		return NamedTuple(std::forward<Field<_name1, Type1>>(_f1),
						  std::forward<Field<_name2, Type2>>(_f2));
	}

	template <StringLiteral _name, class Type, class... FieldTypes>
	inline auto operator*(NamedTuple<FieldTypes...>&& _tup,
						  Field<_name, Type>&& _f) {
		return _tup.add(std::forward<Field<_name, Type>>(_f));
	}

	template <StringLiteral _name, class Type, class... FieldTypes>
	inline auto operator*(Field<_name, Type>&& _f,
						  NamedTuple<FieldTypes...>&& _tup) {
		return NamedTuple(std::forward<Field<_name, Type>>(_f))
				.add(std::forward<NamedTuple<FieldTypes...>>(_tup));
	}

	template <class... FieldTypes1, class... FieldTypes2>
	inline auto operator*(NamedTuple<FieldTypes1...>&& _tup1,
						  NamedTuple<FieldTypes2...>&& _tup2) {
		return _tup1.add(std::forward<NamedTuple<FieldTypes2...>>(_tup2));
	}


	namespace internal {

	}  // namespace internal

	template <class T>
	struct remove_ptr;

	template <StringLiteral _name, class T>
	struct remove_ptr<Field<_name, T>> {
		using FieldType =
		        Field<_name, wrap_in_rfl_array_t<std::remove_cvref_t<std::remove_pointer_t<T>>>>;
	};

	template <class T>
	struct remove_ptrs_nt;

	template <class... FieldTypes>
	struct remove_ptrs_nt<NamedTuple<FieldTypes...>> {
		using NamedTupleType =
		        NamedTuple<typename remove_ptr<FieldTypes>::FieldType...>;
	};

	/// Generates the named tuple that is equivalent to the struct T.
	/// This is the result you would expect from calling to_named_tuple(my_struct).
	/// All fields of the struct must be an Field.
	template <class T>
	using named_tuple_t = typename remove_ptrs_nt<internal::ptr_named_tuple_t<T>>::NamedTupleType;


	template <class T>
	constexpr auto to_ptr_tuple(T& _t) {
		if constexpr (std::is_pointer_v<std::remove_cvref_t<T>>) {
			return to_ptr_tuple(*_t);
		} else {
			return bind_to_tuple(_t, [](auto&& x) {
				return std::addressof(std::forward<decltype(x)>(x));
			});
		}
	}

	template <class T>
	using ptr_tuple_t = decltype(to_ptr_tuple(std::declval<T&>()));


	template <class T>
	struct remove_ptrs_tup;

	template <class... Ts>
	struct remove_ptrs_tup<std::tuple<Ts...>> {
		using TupleType =
		        std::tuple<std::remove_cvref_t<std::remove_pointer_t<Ts>>...>;
	};

	template <class T>
	using tuple_t = typename remove_ptrs_tup<ptr_tuple_t<T>>::TupleType;



	template <class Tuple, int _i = 0>
	constexpr int calc_flattened_size() {
		if constexpr (_i == std::tuple_size_v<Tuple>) {
			return 0;
		} else {
			using T = std::remove_pointer_t<std::tuple_element_t<_i, Tuple>>;
			if constexpr (internal::is_flatten_field_v<T>) {
				return calc_flattened_size<ptr_tuple_t<typename T::Type>>() +
					   calc_flattened_size<Tuple, _i + 1>();
			} else {
				return 1 + calc_flattened_size<Tuple, _i + 1>();
			}
		}
	}

	template <class TargetTupleType, class PtrTupleType, int _j = 0, class... Args>
	constexpr auto unflatten_ptr_tuple(PtrTupleType& _t, Args... _args) noexcept {
		constexpr auto i = sizeof...(Args);

		constexpr auto size = std::tuple_size_v<std::remove_cvref_t<TargetTupleType>>;

		if constexpr (i == size) {
			return std::make_tuple(_args...);
		} else {
			using T = std::remove_cvref_t<
					std::remove_pointer_t<std::tuple_element_t<i, TargetTupleType>>>;

			if constexpr (internal::is_flatten_field_v<T>) {
				using SubTargetTupleType =
						ptr_tuple_t<std::remove_pointer_t<typename T::Type>>;

				constexpr int flattened_size = calc_flattened_size<SubTargetTupleType>();

				return unflatten_ptr_tuple<TargetTupleType, PtrTupleType,
										   _j + flattened_size>(
						_t, _args...,
						unflatten_ptr_tuple<SubTargetTupleType, PtrTupleType, _j>(_t));

			} else {
				return unflatten_ptr_tuple<TargetTupleType, PtrTupleType, _j + 1>(
						_t, _args..., std::get<_j>(_t));
			}
		}
	}

	template <class T, class Pointers, class... Args>
	constexpr auto move_from_pointers(Pointers& _ptrs, Args&&... _args) noexcept {
		constexpr auto i = sizeof...(Args);
		if constexpr (i == std::tuple_size_v<std::remove_cvref_t<Pointers>>) {
			return std::remove_cvref_t<T>{std::move(_args)...};
		} else {
			using FieldType = std::tuple_element_t<i, std::remove_cvref_t<Pointers>>;

			if constexpr (std::is_pointer_v<FieldType>) {
				return move_from_pointers<T>(_ptrs, std::move(_args)...,
											 std::move(*std::get<i>(_ptrs)));

			} else {
				using PtrTupleType = ptr_tuple_t<std::remove_cvref_t<T>>;

				using U = std::remove_cvref_t<typename std::remove_pointer_t<
						typename std::tuple_element_t<i, PtrTupleType>>::Type>;

				return move_from_pointers<T>(_ptrs, std::move(_args)...,
											 move_from_pointers<U>(std::get<i>(_ptrs)));
			}
		}
	}

	template <class T>
	constexpr auto flatten_array(T* _v) noexcept {
		return std::make_tuple(_v);
	}

	template <class T, std::size_t _n>
	constexpr auto flatten_array(std::array<T, _n>* _arr) noexcept {
		const auto fct = [](auto&... _v) {
			return std::tuple_cat(flatten_array(&_v)...);
		};
		return std::apply(fct, *_arr);
	}

	template <class T>
	constexpr auto make_tuple_from_element(T _v) noexcept {
		return std::make_tuple(_v);
	}

	template <class T>
	constexpr auto make_tuple_from_element(Array<T>* _arr) noexcept {
		return flatten_array(&(_arr->arr_));
	}

	constexpr auto flatten_c_arrays(const auto& _tup) noexcept {
		const auto fct = [](auto... _v) {
			return std::tuple_cat(make_tuple_from_element(_v)...);
		};
		return std::apply(fct, _tup);
	}

	/// Generates a named tuple that contains pointers to the original values in
	/// the struct from a tuple.
	template <class TupleType, class... AlreadyExtracted>
	constexpr auto tup_to_ptr_tuple(TupleType& _t, AlreadyExtracted... _a) noexcept {
		constexpr auto i = sizeof...(AlreadyExtracted);
		constexpr auto size = std::tuple_size_v<TupleType>;

		if constexpr (i == size) {
			return std::make_tuple(_a...);
		} else {
			return tup_to_ptr_tuple(_t, _a..., &std::get<i>(_t));
		}
	}

	/// Creates a struct of type T from a tuple by moving the underlying
	/// fields.
	template <class T, class TupleType>
	constexpr auto move_from_tuple(TupleType&& _t) noexcept {
		auto ptr_tuple = tup_to_ptr_tuple(_t);

		using TargetTupleType = tuple_t<std::remove_cvref_t<T>>;

		auto pointers =
				flatten_c_arrays(unflatten_ptr_tuple<TargetTupleType>(ptr_tuple));

		return move_from_pointers<T>(pointers);
	}


	/// Generates the named tuple that is equivalent to the struct _t.
	/// If _t already is a named tuple, then _t will be returned.
	/// All fields of the struct must be an rfl::Field.
	template <class T>
	constexpr auto to_named_tuple(T&& _t) noexcept{
		if constexpr (internal::is_named_tuple_v<std::remove_cvref_t<T>>) {
			return _t;
		} else if constexpr (internal::is_field_v<std::remove_cvref_t<T>>) {
			return make_named_tuple(std::forward<T>(_t));
		} else if constexpr (std::is_lvalue_reference<T>{}) {
			auto field_tuple = internal::copy_to_field_tuple(_t);
			return move_field_tuple_to_named_tuple(std::move(field_tuple));
		} else {
			auto field_tuple = internal::move_to_field_tuple(_t);
			return move_field_tuple_to_named_tuple(std::move(field_tuple));
		}
	}

	/// Generates the named tuple that is equivalent to the struct _t.
	/// If _t already is a named tuple, then _t will be returned.
	/// All fields of the struct must be an rfl::Field.
	template <class T>
	constexpr auto to_named_tuple(const T& _t) noexcept {
		if constexpr (internal::is_named_tuple_v<std::remove_cvref_t<T>>) {
			return _t;
		} else if constexpr (internal::is_field_v<std::remove_cvref_t<T>>) {
			return make_named_tuple(_t);
		} else {
			auto field_tuple = internal::copy_to_field_tuple(_t);
			return move_field_tuple_to_named_tuple(std::move(field_tuple));
		}
	}

	/// Generates the struct T from a named tuple.
	template <class T, class NamedTupleType>
	constexpr auto from_named_tuple(NamedTupleType&& _n) noexcept{
		using RequiredType = std::remove_cvref_t<named_tuple_t<T>>;
		if constexpr (!std::is_same<std::remove_cvref_t<NamedTupleType>,
				RequiredType>()) {
			return from_named_tuple<T>(RequiredType(std::forward<NamedTupleType>(_n)));
		} else if constexpr (internal::has_fields<T>()) {
			if constexpr (std::is_lvalue_reference<NamedTupleType>{}) {
				return copy_from_named_tuple<T>(_n);
			} else {
				return move_from_named_tuple<T>(_n);
			}
		} else {
			if constexpr (std::is_lvalue_reference<NamedTupleType>{}) {
				return copy_from_tuple<T>(_n.values());
			} else {
				return move_from_tuple<T>(std::move(_n.values()));
			}
		}
	}

	/// Generates the struct T from a named tuple.
	template <class T, class NamedTupleType>
	constexpr auto from_named_tuple(const NamedTupleType& _n) noexcept {
		using RequiredType = std::remove_cvref_t<named_tuple_t<T>>;
		if constexpr (!std::is_same<std::remove_cvref_t<NamedTupleType>,
				RequiredType>()) {
			return from_named_tuple<T>(RequiredType(_n));
		} else if constexpr (internal::has_fields<T>()) {
			return internal::copy_from_named_tuple<T>(_n);
		} else {
			return internal::copy_from_tuple<T>(_n.values());
		}
	}



	/// Returns meta-information about the fields.
	template<class T>
	constexpr auto fields() noexcept {
		return internal::get_meta_fields<named_tuple_t<T>>();
	}

	template <class T>
	constexpr auto to_view(T& _t) noexcept {
		return internal::to_ptr_named_tuple(_t);
	}

	/// Replaces one or several fields, returning a new version
	/// with the non-replaced fields left unchanged.
	template <class T, class RField, class... OtherRFields>
	constexpr auto replace(T&& _t, RField&& _field, OtherRFields&&... _other_fields) noexcept {
		if constexpr (internal::is_named_tuple_v<T>) {
			return std::forward<T>(_t).replace(
			        to_named_tuple(std::forward<RField>(_field)),
			        to_named_tuple(std::forward<OtherRFields>(_other_fields))...);
		} else {
			return from_named_tuple<T>(
			        to_named_tuple(std::forward<T>(_t))
			                .replace(to_named_tuple(std::forward<RField>(_field)),
			                         to_named_tuple(
			                                 std::forward<OtherRFields>(_other_fields))...));
		}
	}

	/// Replaces one or several fields, returning a new version
	/// with the non-replaced fields left unchanged.
	template <class T, class RField, class... OtherRFields>
	constexpr auto replace(const T& _t, RField&& _field, OtherRFields&&... _other_fields) noexcept {
		if constexpr (internal::is_named_tuple_v<T>) {
			return _t.replace(
			        to_named_tuple(std::forward<RField>(_field)),
			        to_named_tuple(std::forward<OtherRFields>(_other_fields))...);
		} else {
			return from_named_tuple<T>(to_named_tuple(_t).replace(
			        to_named_tuple(std::forward<RField>(_field)),
			        to_named_tuple(std::forward<OtherRFields>(_other_fields))...));
		}
	}

	/// Generates a type T from the input values.
	template <class T, class Head, class... Tail>
	constexpr T as(Head&& _head, Tail&&... _tail) noexcept {
		if constexpr (sizeof...(_tail) == 0) {
			return from_named_tuple<T>(to_named_tuple(std::forward<Head>(_head)));
		} else {
			return from_named_tuple<T>(
			        to_named_tuple(std::forward<Head>(_head))
			                .add(to_named_tuple(std::forward<Tail>(_tail))...));
		}
	}

	/// Generates a type T from the input values.
	template <class T, class Head, class... Tail>
	constexpr T as(const Head& _head, const Tail&... _tail) noexcept {
		if constexpr (sizeof...(_tail) == 0) {
			return from_named_tuple<T>(to_named_tuple(_head));
		} else {
			return from_named_tuple<T>(
			        to_named_tuple(_head).add(to_named_tuple(_tail)...));
		}
	}

}
#endif//CRYPTANALYSISLIB_REFLECTION_H
