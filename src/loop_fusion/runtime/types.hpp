#pragma once

#include "loop_fusion/runtime/looper.hpp"

#include <variant>

namespace loop_fusion::runtime::types {

/**
 * Backported from C++20.
 * Simply defines the passed in type as `type`.
 */
template <typename T>
struct type_identity {
    using type = T;
};

/**
 * Helper type for generating a variant from a set of types without including
 * a type more than once.
 */
template <typename T, typename... Ts>
struct unique_variant : type_identity<T> {};
template <typename... Ts, typename U, typename... Us>
struct unique_variant<std::variant<Ts...>, U, Us...>
    : std::conditional_t<(std::is_same_v<U, Ts> || ...), unique_variant<std::variant<Ts...>, Us...>,
          unique_variant<std::variant<Ts..., U>, Us...>> {};
template <typename... Ts>
using unique_variant_t = typename unique_variant<std::variant<>, Ts...>::type;

/**
 * Helper type for adding new function types to a looper.
 */
template <typename Looper, typename... Fs>
struct looper_append;
template <typename... FsOld, typename... FsNew>
struct looper_append<looper<FsOld...>, FsNew...> {
    using type = looper<FsOld..., FsNew...>;
};
template <typename Looper, typename... Fs>
using looper_append_t = typename looper_append<Looper, Fs...>::type;

/**
 * Helper type for adding new function types to all loopers in a variant.
 * It preserves the variant's existing types, adds new versions of them with the
 * appended function types and also adds loopers that only contain the new function
 * types.
 * Duplicate looper types are removed from the result through `unique_variant_t`.
 */
template <typename V, typename... Fs>
struct variant_append;
template <typename... Ls, typename... Fs>
struct variant_append<std::variant<Ls...>, Fs...> {
    using type = unique_variant_t<Ls..., looper_append_t<Ls, Fs...>..., looper<Fs...>>;
};
template <typename V, typename... Fs>
using variant_append_t = typename variant_append<V, Fs...>::type;

/**
 * Extension of `variant_append_t` to add all functions of a looper to
 * a variant.
 */
template <typename V, typename Looper>
struct variant_append_looper;
template <typename... A, typename... T>
struct variant_append_looper<std::variant<A...>, looper<T...>> : variant_append<std::variant<A...>, T...> {};
template <typename V, typename Looper>
using variant_append_looper_t = typename variant_append<V, Looper>::type;

/**
 * Helper type for merging multiple variants into one, filtering out any
 * duplicate types through `unique_variant_t`.
 */
template <typename V, typename... Vs>
struct variant_merge : type_identity<V> {};
template <typename... A, typename... B, typename... Vs>
struct variant_merge<std::variant<A...>, std::variant<B...>, Vs...>
    : variant_merge<unique_variant_t<A..., B...>, Vs...> {};
template <typename... Vs>
using variant_merge_t = typename variant_merge<Vs...>::type;

/**
 * Helper type for appending the function types from a set of looper types
 * to the loopers contained in a variant.
 */
template <typename V, typename... Ls>
using variant_append_all_t = variant_merge_t<variant_append_looper_t<V, Ls>...>;

} // namespace loop_fusion::runtime::types
