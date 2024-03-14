#ifndef CRYPTANALYSISLIB_REFLECTION_NAME_TUPLE_H
#define CRYPTANALYSISLIB_REFLECTION_NAME_TUPLE_H

#include "reflection/internal.h"


namespace cryptanalysislib::reflection {
	/// A named tuple behaves like std::tuple,
	/// but the fields have explicit names, which
	/// allows for reflection.
	/// IMPORTANT: We have two template specializations. One with fields, one
	/// without fields.
	template<class... FieldTypes>
	class NamedTuple;

	// ----------------------------------------------------------------------------

	template<class... FieldTypes>
	class NamedTuple {
	public:
		using Fields = std::tuple<std::remove_cvref_t<FieldTypes>...>;
		using Values =
		        std::tuple<typename std::remove_cvref<FieldTypes>::type::Type...>;

	public:
		/// Construct from the values.
		constexpr NamedTuple(typename std::remove_cvref<FieldTypes>::type::Type &&..._values) noexcept
		    : values_(std::forward<typename std::remove_cvref<FieldTypes>::type::Type>(_values)...) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Construct from the values.
		constexpr NamedTuple(const typename std::remove_cvref<FieldTypes>::type::Type &..._values) noexcept
		    : values_(std::make_tuple(_values...)) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Construct from the fields.
		constexpr NamedTuple(FieldTypes &&..._fields) noexcept
		    : values_(std::make_tuple(std::move(_fields.value_)...)) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Construct from the fields.
		constexpr NamedTuple(const FieldTypes &..._fields) noexcept
		    : values_(std::make_tuple(_fields.value_...)) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Construct from a tuple containing fields.
		constexpr NamedTuple(std::tuple<FieldTypes...> &&_tup) noexcept
		    : NamedTuple(std::make_from_tuple<NamedTuple<FieldTypes...>>(
		              std::forward<std::tuple<FieldTypes...>>(_tup))) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Construct from a tuple containing fields.
		constexpr NamedTuple(const std::tuple<FieldTypes...> &_tup) noexcept
		    : NamedTuple(std::make_from_tuple<NamedTuple<FieldTypes...>>(_tup)) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Copy constructor.
		constexpr NamedTuple(const NamedTuple<FieldTypes...> &_other) = default;

		/// Move constructor.
		constexpr NamedTuple(NamedTuple<FieldTypes...> &&_other) = default;

		/// Copy constructor.
		template<class... OtherFieldTypes>
		constexpr NamedTuple(const NamedTuple<OtherFieldTypes...> &_other) noexcept
		    : NamedTuple(retrieve_fields(_other.fields())) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		/// Move constructor.
		template<class... OtherFieldTypes>
		constexpr NamedTuple(NamedTuple<OtherFieldTypes...> &&_other) noexcept
		    : NamedTuple(retrieve_fields(_other.fields())) {
			static_assert(no_duplicate_field_names(),
			              "Duplicate field names are not allowed");
		}

		constexpr ~NamedTuple() = default;

		/// Returns a new named tuple with additional fields.
		template<class Head, class... Tail>
		constexpr auto add(Head &&_head, Tail &&..._tail) noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return NamedTuple<FieldTypes..., std::remove_cvref_t<Head>>(
				               make_fields<1, Head>(std::forward<Head>(_head)))
				        .add(std::forward<Tail>(_tail)...);
			} else {
				return NamedTuple<FieldTypes..., std::remove_cvref_t<Head>>(
				        make_fields<1, Head>(std::forward<Head>(_head)));
			}
		}

		/// Returns a new named tuple with additional fields.
		template<class Head, class... Tail>
		constexpr auto add(Head &&_head, Tail &&..._tail) const noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return NamedTuple<FieldTypes..., std::remove_cvref_t<Head>>(
				               make_fields<1, Head>(std::forward<Head>(_head)))
				        .add(std::forward<Tail>(_tail)...);
			} else {
				return NamedTuple<FieldTypes..., std::remove_cvref_t<Head>>(
				        make_fields<1, Head>(std::forward<Head>(_head)));
			}
		}

		/// Template specialization for std::tuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto add(std::tuple<TupContent...> &&_tuple, Tail &&..._tail) noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return add_tuple(std::forward<std::tuple<TupContent...>>(_tuple))
				        .add(std::forward<Tail>(_tail)...);
			} else {
				return add_tuple(std::forward<std::tuple<TupContent...>>(_tuple));
			}
		}

		/// Template specialization for std::tuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto add(std::tuple<TupContent...> &&_tuple, Tail &&..._tail) const noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return add_tuple(std::forward<std::tuple<TupContent...>>(_tuple))
				        .add(std::forward<Tail>(_tail)...);
			} else {
				return add_tuple(std::forward<std::tuple<TupContent...>>(_tuple));
			}
		}

		/// Template specialization for NamedTuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto add(NamedTuple<TupContent...> &&_named_tuple, Tail &&..._tail) noexcept {
			return add(std::forward<std::tuple<TupContent...>>(_named_tuple.fields()),
			           std::forward<Tail>(_tail)...);
		}

		/// Template specialization for NamedTuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto add(NamedTuple<TupContent...> &&_named_tuple, Tail &&..._tail) const noexcept {
			return add(std::forward<std::tuple<TupContent...>>(_named_tuple.fields()),
			           std::forward<Tail>(_tail)...);
		}

		/// Creates a new named tuple by applying the supplied function to
		/// field. The function is expected to return a named tuple itself.
		template<typename F>
		constexpr auto and_then(const F &_f) noexcept {
			const auto transform_field = [&_f](auto... _fields) {
				return std::tuple_cat(_f(std::move(_fields)).fields()...);
			};
			const auto to_nt = []<class... NewFields>(std::tuple<NewFields...> &&_tup) {
				return NamedTuple<NewFields...>(_tup);
			};
			auto new_fields = std::apply(transform_field, std::move(fields()));
			return to_nt(std::move(new_fields));
		}

		/// Creates a new named tuple by applying the supplied function to
		/// field. The function is expected to return a named tuple itself.
		template<typename F>
		constexpr auto and_then(const F &_f) const noexcept {
			const auto transform_field = [&_f](auto... _fields) {
				return std::tuple_cat(_f(std::move(_fields)).fields()...);
			};
			const auto to_nt = []<class... NewFields>(std::tuple<NewFields...> &&_tup) {
				return NamedTuple<NewFields...>(_tup);
			};
			auto new_fields = std::apply(transform_field, std::move(fields()));
			return to_nt(std::move(new_fields));
		}

		/// Invokes a callable object once for each field in order.
		template<typename F>
		constexpr void apply(const F &_f) noexcept {
			const auto apply_to_field =
			        [&_f]<typename... AFields>(AFields &&...fields) {
				        ((_f(std::forward<AFields>(fields))), ...);
			        };
			std::apply(apply_to_field, fields());
		}

		/// Invokes a callable object once for each field in order.
		template<typename F>
		constexpr void apply(const F &_f) const noexcept {
			const auto apply_to_field = [&_f](const auto &...fields) {
				((_f(fields)), ...);
			};
			std::apply(apply_to_field, fields());
		}

		/// Returns a tuple containing the fields.
		constexpr Fields fields() noexcept { return make_fields(); }

		/// Returns a tuple containing the fields.
		constexpr Fields fields() const noexcept { return make_fields(); }

		/// Gets a field by index.
		template<int _index>
		constexpr auto &get() noexcept {
			return get<_index>(*this);
		}

		/// Gets a field by name.
		template<StringLiteral _field_name>
		constexpr auto &get() noexcept {
			return get<_field_name>(*this);
		}

		/// Gets a field by the field type.
		template<class Field>
		constexpr auto &get() noexcept{
			return get<Field>(*this);
		}

		/// Gets a field by index.
		template<int _index>
		constexpr const auto &get() const noexcept {
			return get<_index>(*this);
		}

		/// Gets a field by name.
		template<StringLiteral _field_name>
		constexpr const auto &get() const noexcept {
			return get<_field_name>(*this);
		}

		/// Gets a field by the field type.
		template<class Field>
		constexpr const auto &get() const noexcept {
			return get<Field>(*this);
		}

		/// Returns the results wrapped in a field.
		template<StringLiteral _field_name>
		constexpr auto get_field() const noexcept {
			return make_field<_field_name>(get<_field_name>(*this));
		}

		/// Copy assignment operator.
		constexpr NamedTuple<FieldTypes...> &operator=(
		        const NamedTuple<FieldTypes...> &_other) = default;

		/// Move assignment operator.
		constexpr NamedTuple<FieldTypes...> &operator=(
		        NamedTuple<FieldTypes...> &&_other) noexcept = default;

		/// Replaces one or several fields, returning a new version
		/// with the non-replaced fields left unchanged.
		template<class RField, class... OtherRFields>
		constexpr NamedTuple<FieldTypes...> replace(RField &&_field,
		                                  OtherRFields &&..._other_fields) noexcept {
			constexpr auto num_other_fields = sizeof...(OtherRFields);
			if constexpr (num_other_fields == 0) {
				return replace_value<RField>(_field.value_);
			} else {
				return replace_value<RField>(_field.value_)
				        .replace(std::forward<OtherRFields>(_other_fields)...);
			}
		}

		/// Replaces one or several fields, returning a new version
		/// with the non-replaced fields left unchanged.
		template<class RField, class... OtherRFields>
		constexpr NamedTuple<FieldTypes...> replace(RField &&_field,
		                                  OtherRFields &&..._other_fields) const noexcept {
			constexpr auto num_other_fields = sizeof...(OtherRFields);
			if constexpr (num_other_fields == 0) {
				return replace_value<RField>(_field.value_);
			} else {
				return replace_value<RField>(_field.value_)
				        .replace(std::forward<OtherRFields>(_other_fields)...);
			}
		}

		/// Template specialization for std::tuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto replace(std::tuple<TupContent...> &&_tuple, Tail &&..._tail) noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return replace_tuple(std::forward<std::tuple<TupContent...>>(_tuple))
				        .replace(std::forward<Tail>(_tail)...);
			} else {
				return replace_tuple(std::forward<std::tuple<TupContent...>>(_tuple));
			}
		}

		/// Template specialization for std::tuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto replace(std::tuple<TupContent...> &&_tuple, Tail &&..._tail) const noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return replace_tuple(std::forward<std::tuple<TupContent...>>(_tuple))
				        .replace(std::forward<Tail>(_tail)...);
			} else {
				return replace_tuple(std::forward<std::tuple<TupContent...>>(_tuple));
			}
		}

		/// Template specialization for NamedTuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto replace(NamedTuple<TupContent...> &&_named_tuple, Tail &&..._tail) noexcept {
			return replace(
			        std::forward<NamedTuple<TupContent...>>(_named_tuple).fields(),
			        std::forward<Tail>(_tail)...);
		}

		/// Template specialization for NamedTuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto replace(const NamedTuple<TupContent...> &_named_tuple,
		             Tail &&..._tail) const noexcept {
			return replace(
			        std::forward<NamedTuple<TupContent...>>(_named_tuple).fields(),
			        std::forward<Tail>(_tail)...);
		}

		/// Returns the size of the named tuple
		static constexpr size_t size() noexcept { return std::tuple_size_v<Values>; }

		/// Creates a new named tuple by applying the supplied function to every
		/// field.
		template<typename F>
		constexpr auto transform(const F &_f) noexcept {
			const auto transform_field = [&_f](auto... fields) {
				return std::make_tuple(_f(std::move(fields))...);
			};
			const auto to_nt = []<class... NewFields>(std::tuple<NewFields...> &&_tup) {
				return NamedTuple<NewFields...>(_tup);
			};
			auto new_fields = std::apply(transform_field, std::move(fields()));
			return to_nt(std::move(new_fields));
		}

		/// Creates a new named tuple by applying the supplied function to every
		/// field.
		template<typename F>
		constexpr auto transform(const F &_f) const noexcept {
			const auto transform_field = [&_f](auto... fields) {
				return std::make_tuple(_f(std::move(fields))...);
			};
			const auto to_nt = []<class... NewFields>(std::tuple<NewFields...> &&_tup) {
				return NamedTuple<NewFields...>(_tup);
			};
			auto new_fields = std::apply(transform_field, std::move(fields()));
			return to_nt(std::move(new_fields));
		}

		/// Returns the underlying std::tuple.
		constexpr Values &values() noexcept { return values_; }

		/// Returns the underlying std::tuple.
		constexpr const Values &values() const noexcept { return values_; }

	private:
		/// Adds the elements of a tuple to a newly created named tuple,
		/// and other elements to a newly created named tuple.
		template<class... TupContent>
		constexpr auto add_tuple(std::tuple<TupContent...> &&_tuple) noexcept {
			const auto a = [this](auto &&..._fields) {
				return this->add(std::forward<TupContent>(_fields)...);
			};
			return std::apply(a, std::forward<std::tuple<TupContent...>>(_tuple));
		}

		/// Adds the elements of a tuple to a newly created named tuple,
		/// and other elements to a newly created named tuple.
		template<class... TupContent>
		constexpr auto add_tuple(std::tuple<TupContent...> &&_tuple) const noexcept {
			const auto a = [this](auto &&..._fields) {
				return this->add(std::forward<TupContent>(_fields)...);
			};
			return std::apply(a, std::forward<std::tuple<TupContent...>>(_tuple));
		}

		/// Generates the fields.
		template<int num_additional_fields = 0, class... Args>
		constexpr auto make_fields(Args &&..._args) noexcept {
			constexpr auto size = sizeof...(Args) - num_additional_fields;
			constexpr auto num_fields = std::tuple_size_v<Fields>;
			constexpr auto i = num_fields - size - 1;

			constexpr bool retrieved_all_fields = size == num_fields;

			if constexpr (retrieved_all_fields) {
				return std::make_tuple(std::forward<Args>(_args)...);
			} else {
				// When we add additional fields, it is more intuitive to add
				// them to the end, that is why we do it like this.
				using FieldType = typename std::tuple_element<i, Fields>::type;
				using T = std::remove_cvref_t<typename FieldType::Type>;
				return make_fields<num_additional_fields>(
				        FieldType(std::forward<T>(std::get<i>(values_))),
				        std::forward<Args>(_args)...);
			}
		}

		/// Generates the fields.
		template<int num_additional_fields = 0, class... Args>
		constexpr auto make_fields(Args &&..._args) const noexcept {
			constexpr auto size = sizeof...(Args) - num_additional_fields;
			constexpr auto num_fields = std::tuple_size_v<Fields>;
			constexpr auto i = num_fields - size - 1;

			constexpr bool retrieved_all_fields = size == num_fields;

			if constexpr (retrieved_all_fields) {
				return std::make_tuple(std::forward<Args>(_args)...);
			} else {
				// When we add additional fields, it is more intuitive to add
				// them to the end, that is why we do it like this.
				using FieldType = typename std::tuple_element<i, Fields>::type;
				return make_fields<num_additional_fields>(FieldType(std::get<i>(values_)),
				                                          std::forward<Args>(_args)...);
			}
		}

		/// Generates a new named tuple with one value replaced with a new value.
		template<int _index, class V, class T, class... Args>
		constexpr auto make_replaced(V &&_values, T &&_val, Args &&..._args) const noexcept {
			constexpr auto size = sizeof...(Args);

			constexpr bool retrieved_all_fields = size == std::tuple_size_v<Fields>;

			if constexpr (retrieved_all_fields) {
				return NamedTuple<FieldTypes...>(std::forward<Args>(_args)...);
			} else {
				using FieldType = typename std::tuple_element<size, Fields>::type;

				if constexpr (size == _index) {
					return make_replaced<_index, V, T>(
					        std::forward<V>(_values), std::forward<T>(_val),
					        std::forward<Args>(_args)..., FieldType(std::forward<T>(_val)));
				} else {
					using U = typename FieldType::Type;
					return make_replaced<_index, V, T>(
					        std::forward<V>(_values), std::forward<T>(_val),
					        std::forward<Args>(_args)...,
					        FieldType(std::forward<U>(std::get<size>(_values))));
				}
			}
		}

		/// We cannot allow duplicate field names.
		constexpr static bool no_duplicate_field_names() noexcept {
			return no_duplicate_field_names_<Fields>();
		}

		/// Replaced the field signified by the field type.
		template<class Field, class T>
		constexpr NamedTuple<FieldTypes...> replace_value(T &&_val) noexcept {
			using FieldType = std::remove_cvref_t<Field>;
			constexpr auto index = find_index<FieldType::name_, Fields>();
			return make_replaced<index, Values, T>(std::forward<Values>(values_),
			                                       std::forward<T>(_val));
		}

		/// Replaced the field signified by the field type.
		template<class Field, class T>
		constexpr NamedTuple<FieldTypes...> replace_value(T &&_val) const noexcept {
			using FieldType = std::remove_cvref_t<Field>;
			constexpr auto index = find_index<FieldType::name_, Fields>();
			auto values = values_;
			return make_replaced<index, Values, T>(std::move(values),
			                                       std::forward<T>(_val));
		}

		/// Adds the elements of a tuple to a newly created named tuple,
		/// and other elements to a newly created named tuple.
		template<class... TupContent>
		constexpr auto replace_tuple(std::tuple<TupContent...> &&_tuple) noexcept {
			const auto r = [this](auto &&..._fields) {
				return this->replace(std::forward<TupContent>(_fields)...);
			};
			return std::apply(r, std::forward<std::tuple<TupContent...>>(_tuple));
		}

		/// Adds the elements of a tuple to a newly created named tuple,
		/// and other elements to a newly created named tuple.
		template<class... TupContent>
		constexpr auto replace_tuple(std::tuple<TupContent...> &&_tuple) const noexcept {
			const auto r = [this](auto &&..._fields) {
				return this->replace(std::forward<TupContent>(_fields)...);
			};
			return std::apply(r, std::forward<std::tuple<TupContent...>>(_tuple));
		}

		/// Retrieves the fields from another tuple.
		template<class... OtherFieldTypes, class... Args>
		constexpr static Fields retrieve_fields(
		        std::tuple<OtherFieldTypes...> &&_other_fields, Args &&..._args) noexcept {
			constexpr auto size = sizeof...(Args);

			constexpr bool retrieved_all_fields = size == std::tuple_size_v<Fields>;

			if constexpr (retrieved_all_fields) {
				return std::make_tuple(std::forward<Args>(_args)...);
			} else {
				constexpr auto field_name = std::tuple_element<size, Fields>::type::name_;
				constexpr auto index = find_index<field_name, std::tuple<OtherFieldTypes...>>();
				using FieldType = typename std::tuple_element<size, Fields>::type;
				using T = std::remove_cvref_t<typename FieldType::Type>;

				return retrieve_fields(
				        std::forward<std::tuple<OtherFieldTypes...>>(_other_fields),
				        std::forward<Args>(_args)...,
				        FieldType(std::forward<T>(std::get<index>(_other_fields).value_)));
			}
		}

	private:
		/// The values actually contained in the named tuple.
		/// As you can see, a NamedTuple is just a normal tuple under-the-hood,
		/// everything else is resolved at compile time. It should have no
		/// runtime overhead over a normal std::tuple.
		Values values_;
	};

	// ----------------------------------------------------------------------------

	/// We need a special template instantiation for empty named tuples.
	template<>
	class NamedTuple<> {
	public:
		using Fields = std::tuple<>;
		using Values = std::tuple<>;

		constexpr NamedTuple(){};
		constexpr ~NamedTuple() = default;

		/// Returns a new named tuple with additional fields.
		template<class Head, class... Tail>
		constexpr auto add(Head &&_head, Tail &&..._tail) const noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return NamedTuple<std::remove_cvref_t<Head>>(std::forward<Head>(_head))
				        .add(std::forward<Tail>(_tail)...);
			} else {
				return NamedTuple<std::remove_cvref_t<Head>>(std::forward<Head>(_head));
			}
		}

		/// Template specialization for std::tuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto add(std::tuple<TupContent...> &&_tuple, Tail &&..._tail) const noexcept {
			if constexpr (sizeof...(Tail) > 0) {
				return NamedTuple<TupContent...>(
				               std::forward<std::tuple<TupContent...>>(_tuple))
				        .add(std::forward<Tail>(_tail)...);
			} else {
				return NamedTuple<TupContent...>(
				        std::forward<std::tuple<TupContent...>>(_tuple));
			}
		}

		/// Template specialization for NamedTuple, so we can pass fields from other
		/// named tuples.
		template<class... TupContent, class... Tail>
		constexpr auto add(NamedTuple<TupContent...> &&_named_tuple, Tail &&..._tail) const noexcept {
			return add(std::forward<std::tuple<TupContent...>>(_named_tuple.fields()),
			           std::forward<Tail>(_tail)...);
		}

		/// Returns an empty named tuple.
		template<typename F>
		constexpr auto and_then(const F &_f) const noexcept {
			(void)_f;
			return NamedTuple<>();
		}

		/// Does nothing at all.
		template<typename F>
		constexpr void apply(const F &_f) const noexcept { (void)_f; }

		/// Returns an empty tuple.
		constexpr auto fields() const noexcept { return std::tuple(); }

		/// Must always be 0.
		static constexpr size_t size() noexcept { return 0; }

		/// Returns an empty named tuple.
		template<typename F>
		constexpr auto transform(const F &_f) const noexcept {
			(void)_f;
			return NamedTuple<>();
		}

		/// Returns an empty tuple.
		constexpr auto values() const noexcept { return std::tuple(); }
	};
}
#endif//CRYPTANALYSISLIB_NAME_TUPLE_H
