#pragma once

#include <tuple>
#include <type_traits>

namespace loop_fusion::compiletime::types {

template <typename T, typename... U>
using all_same = std::conjunction<std::is_same<T, U>...>;

// Tuple ---------------------------------------------------

template <class ExistingTuple, class New>
struct add_to_tuple;

template <class... ExistingTypes, class New>
struct add_to_tuple<std::tuple<ExistingTypes...>, New> {
    using type = std::tuple<ExistingTypes..., New>;
};
template <class ExistingTuple, class New>
using add_to_tuple_t = typename add_to_tuple<ExistingTuple, New>::type;

/// Wraps its value in a std::tuple unless the value is itself already
/// a tuple.
template <typename Value>
struct as_tuple {
    using type = std::tuple<Value>;
};
template <typename... Values>
struct as_tuple<std::tuple<Values...>> {
    using type = std::tuple<Values...>;
};
template <typename Value>
using as_tuple_t = typename as_tuple<Value>::type;

/// Flattens a tuple of tuple to not contain tuples itself.
template <class ExistingTuple>
struct flatten_tuple;

template <class... Values>
struct flatten_tuple<std::tuple<Values...>> {
    using type = decltype(std::tuple_cat(as_tuple_t<Values>()...));
};
template <typename ExistingTuple>
using flatten_tuple_t = typename flatten_tuple<ExistingTuple>::type;

// Integer sequence helpers ---------------------------------------------------

/// Increases all values of the index_sequence by a value of "Add"
/// For example: <1,2,3,4> + 2 => <3,4,5,6>
template <class IndexSequence, std::size_t Add>
struct increase_index_sequence;

template <std::size_t... I, std::size_t Add>
struct increase_index_sequence<std::integer_sequence<std::size_t, I...>, Add> {
    using type = std::integer_sequence<std::size_t, (I + Add)...>;
};

/// integer sequence of type size_t of the range [Start, End)
template <std::size_t Start, std::size_t End>
using index_sequence_from_to = typename increase_index_sequence<std::make_index_sequence<End - Start>, Start>::type;

namespace details {

    template <class SeqLhs, class SeqRhs>
    struct index_sequence_impl;

    template <std::size_t... ValuesLhs, std::size_t... ValuesRhs>
    struct index_sequence_impl<std::integer_sequence<std::size_t, ValuesLhs...>,
        std::integer_sequence<std::size_t, ValuesRhs...>> {
        using type = std::index_sequence<ValuesLhs..., ValuesRhs...>;
    };

} // namespace details

/// Concatenates two index_sequences to a single one.
template <class SeqLhs, class SeqRhs>
using index_sequence_cat = typename details::index_sequence_impl<SeqLhs, SeqRhs>::type;

} // namespace loop_fusion::compiletime::types
