#pragma once

#include "loop_fusion/compiletime/basic_looper.hpp"
#include "loop_fusion/compiletime/types.hpp"

#include <tuple>
#include <type_traits>

namespace loop_fusion::compiletime {

/// Helper for basic_looper_union. Describes a loop run with start + end values
/// as well as the indexes of the functions that should be run in that loop.
/// Indexes should reference the one of the tuple of functions given to
/// basic_looper_union.
template <typename Iter, Iter Start, Iter End, typename LoopIndexes>
class basic_looper_union_range;

template <typename Iter, Iter Start, Iter End, std::size_t... LoopIndexes>
class basic_looper_union_range<Iter, Start, End, std::index_sequence<LoopIndexes...>> {
public:
    using iterator = Iter;
    static constexpr Iter start { Start };
    static constexpr Iter end { End };
    using indexes_t = std::index_sequence<LoopIndexes...>;
    static constexpr indexes_t indexes {};
};

template <typename Iter, typename T>
class basic_looper_union;

/// A union of loops with different indexes. For a description of how the
/// tuples work together, see MergingLoops.md
///
/// \tparam Iter Iteration Type, e.g. int or size_t
/// \tparam LoopUnionSingle Must be of type basic_looper_union_range
template <typename Iter, typename... LoopUnionSingle, typename... F>
class basic_looper_union<Iter, std::pair<std::tuple<LoopUnionSingle...>, std::tuple<F...>>> {
public:
    // Pre-condition: We only support integral iteration types like std::size_t or int.
    static_assert(std::is_integral_v<Iter>, "loop value type must be an integral value type");
    // All Iter must be the same
    static_assert(types::all_same<Iter, typename LoopUnionSingle::iterator...>::value,
        "Iterator Type of loops must be the same.");

    using self = basic_looper_union<Iter, std::pair<std::tuple<LoopUnionSingle...>, std::tuple<F...>>>;
    using iterator = Iter;
    using loops = std::tuple<LoopUnionSingle...>;
    static constexpr std::size_t function_count = sizeof...(F);

    /// Type of functions that can be used for loop fusion.
    /// Function must accept an argument of type Iter per value.
    template <typename loop_function>
    using is_loop_function = std::is_invocable<loop_function&, Iter>;

public:
    explicit basic_looper_union(std::tuple<F...> _functions)
        : functions { _functions }
    {
        static_assert(std::conjunction_v<is_loop_function<F>...>);
    }

    void run()
    {
        (run_loop<LoopUnionSingle::start, LoopUnionSingle::end>(LoopUnionSingle::indexes), ...);
    }

    std::tuple<F...> functions;

private:
    template <Iter Start, Iter End, std::size_t... Idx>
    void run_loop(std::index_sequence<Idx...>)
    {
        for (auto i = Start; i != End; ++i) {
            (std::get<Idx>(functions)(i), ...);
        }
    }
};

template <typename IterLhs, typename TLhs, class LooperRhs>
auto operator|(basic_looper_union<IterLhs, TLhs> lhs_union, LooperRhs looper_rhs)
{
    return add_to_union(lhs_union, looper_rhs);
}

} // namespace loop_fusion::compiletime
