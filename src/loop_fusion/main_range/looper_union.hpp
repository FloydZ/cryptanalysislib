#pragma once

#include "loop_fusion/common/range.hpp"
#include "loop_fusion/common/types.hpp"
#include "loop_fusion/main_range/looper.hpp"

#include <tuple>

namespace loop_fusion::main_range {

/// Merge the given loopers so that the common range is executed
/// only once. If one or more loopers are disjunct from one another
/// then no merging is performed.
///
/// \tparam Loopers A basic_looper type
/// \param loopers A basic_looper instance
template <class... Loopers>
constexpr void merge_and_run(Loopers... loopers)
{
    const auto all_loopers = std::tuple<Loopers...> { loopers... };
    static_assert(std::tuple_size<decltype(all_loopers)>::value > 0, "cannot merge zero loopers");

    // Initialize start and end. We can use the tuple above for that.
    using Iterator = typename std::tuple_element<0, std::tuple<Loopers...>>::type::iterator;
    Iterator start = std::get<0>(all_loopers).bounds.start;
    Iterator end = std::get<0>(all_loopers).bounds.end;

    const auto createMainSegment = [&start, &end](const auto& singleLooper) {
        if (singleLooper.bounds.start > start) {
            start = singleLooper.bounds.start;
        }
        if (singleLooper.bounds.end < end) {
            end = singleLooper.bounds.end;
        }
    };

    (..., createMainSegment(loopers));

    if (start >= end) {
        // At least one loop does not have a common range
        // with the others. Therefore execute them sequentially.
        (..., loopers.run());
    } else {
        auto function_tuple = std::tuple_cat(loopers.functions...);
        auto main_segment_looper = main_range::looper(common::basic_range<Iterator> { start, end }, function_tuple);
        // We have a common segment and can therefore run all loops
        // from their start to the common start.
        // This only works because no loop is disjunct from another.
        (..., loopers.run(loopers.bounds.start, start));
        main_segment_looper.run();
        (..., loopers.run(end, loopers.bounds.end));
    }
}

/// A union of loopers with different ranges.
///
/// \tparam Iterator Type for the iterator variable. Must be the same for all given loopers.
/// \tparam Loopers  Loopers with different ranges.
template <typename Iterator, class... Loopers>
class looper_union {
public:
    using self = looper_union<Iterator, Loopers...>;
    using iterator = Iterator;

    static_assert(sizeof...(Loopers) > 0, "expected at least one looper in looper_union");
    static_assert(std::conjunction_v<std::is_same<Iterator, typename Loopers::iterator>...>,
        "iterator type of looper must be the same");

public:
    constexpr looper_union(common::type_identity<Iterator>, const std::tuple<Loopers...>& _loopers)
        : loopers(_loopers)
    {}

    /// Merge and run all loops. Only one main intersection is used if one exists.
    constexpr void run()
    {
        run_loops(std::make_index_sequence<sizeof...(Loopers)> {});
    }

    template <typename IteratorRhs, typename... F>
    [[nodiscard]] constexpr auto operator|(looper<IteratorRhs, F...> rhs_looper) const
    {
        static_assert(std::is_same_v<Iterator, IteratorRhs>, "cannot add looper with different iterator type");
        auto looper_tuple = std::tuple_cat(loopers, std::make_tuple(rhs_looper));
        return main_range::looper_union(common::type_identity<Iterator> {}, looper_tuple);
    }

    template <typename IteratorRhs, typename... LoopersRhs>
    [[nodiscard]] constexpr auto operator|(looper_union<IteratorRhs, LoopersRhs...> rhs_union) const
    {
        static_assert(std::is_same_v<Iterator, IteratorRhs>, "cannot add looper_union with different iterator type");
        auto union_tuple = std::tuple_cat(loopers, rhs_union.loopers);
        return main_range::looper_union(common::type_identity<Iterator> {}, union_tuple);
    }

private:
    std::tuple<Loopers...> loopers;

private:
    template <std::size_t... Idx>
    constexpr void run_loops(std::index_sequence<Idx...>)
    {
        merge_and_run(std::get<Idx>(loopers)...);
    }
};

} // namespace loop_fusion::main_range
