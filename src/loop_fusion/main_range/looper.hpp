
#pragma once

#include "loop_fusion/common/range.hpp"
#include "loop_fusion/common/types.hpp"

#include <tuple>
#include <type_traits>

namespace loop_fusion::main_range {

using range = loop_fusion::common::range;

// Forward declaration
template <typename Iterator, class... Loopers>
class looper_union;

/// Base looper class. Represents a loop over a given range. In each
/// iterator all functions F... are executed.
///
/// \tparam Iterator Type for the iterator variable. Typically size_t
/// \tparam F        Lambda functions
template <typename Iterator, typename... F>
class looper {
public:
    using self = looper<Iterator, F...>;
    using iterator = Iterator;
    static constexpr auto function_count = sizeof...(F);

    /// Type of functions that can be used for loop fusion.
    /// Function must accept an argument of type Iter per value.
    template <typename loop_function>
    using is_loop_function = std::is_invocable<loop_function&, Iterator>;

    static_assert(std::conjunction_v<is_loop_function<F>...>, "invalid function signature");
    static_assert(std::is_integral_v<Iterator>, "only integral types are allowed as iterators");

    common::basic_range<Iterator> bounds;
    std::tuple<F...> functions;

public:
    constexpr looper(common::basic_range<Iterator> _range, std::tuple<F...> _functions) noexcept
        : bounds { _range }
        , functions(_functions) {};

    constexpr void run()
    {
        run_loops(std::index_sequence_for<F...> {}, bounds.start, bounds.end);
    }

    constexpr void run(Iterator start, Iterator end)
    {
        run_loops(std::index_sequence_for<F...> {}, start, end);
    }

    template <typename loop_function, typename = std::enable_if_t<is_loop_function<loop_function>::value>>
    [[nodiscard]] constexpr auto operator|(loop_function rhs_function) const
    {
        using new_looper = looper<Iterator, F..., loop_function>;
        return new_looper(bounds, std::tuple_cat(functions, std::make_tuple(rhs_function)));
    }

    template <typename IteratorRhs, typename... F2>
    [[nodiscard]] constexpr auto operator|(looper<IteratorRhs, F2...> rhs) const
    {
        static_assert(std::is_same_v<Iterator, IteratorRhs>, "cannot add looper with different iterator type");
        return looper_union(common::type_identity<Iterator> {}, std::make_tuple(*this, rhs));
    }

private:
    template <std::size_t... Idx>
    constexpr void run_loops(std::index_sequence<Idx...>, Iterator start, Iterator end)
    {
        for (Iterator i = start; i != end; ++i) {
            (std::get<Idx>(functions)(i), ...);
        }
    }
};

} // namespace loop_fusion::main_range
