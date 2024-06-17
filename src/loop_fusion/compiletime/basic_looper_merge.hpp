#pragma once

#include "loop_fusion/compiletime/basic_looper.hpp"
#include "loop_fusion/compiletime/basic_looper_union.hpp"
#include "loop_fusion/compiletime/types.hpp"

#include <tuple>
#include <type_traits>

namespace loop_fusion::compiletime {

/// Merge two basic_looper instances into one basic_looper_union.
template <class LooperLhs, class LooperRhs>
constexpr auto make_union(LooperLhs lhs, LooperRhs rhs) {
    using lhs_part = basic_looper_union_range<std::size_t, LooperLhs::start, LooperLhs::end,
        std::make_index_sequence<LooperLhs::function_count>>;
    using pair = std::pair<std::tuple<lhs_part>, decltype(lhs.functions)>;
    auto looper_union_lhs = basic_looper_union<typename LooperLhs::iterator, pair> { lhs.functions };
    return add_to_union(looper_union_lhs, rhs);
}

template <class LoopRangesTuple, std::size_t Index, class NewRange>
constexpr auto merge_with_range_impl() {
    using LoopRangeOld = std::decay_t<decltype(std::get<Index>(LoopRangesTuple {}))>;
    using iter = typename LoopRangeOld::iterator;
    static_assert(std::is_same_v<iter, typename NewRange::iterator>, "Iterators must be the same");

    using merged_indexes = types::index_sequence_cat<typename LoopRangeOld::indexes_t, typename NewRange::indexes_t>;
    using loop_with_rhs_merged = basic_looper_union_range<iter, LoopRangeOld::start, LoopRangeOld::end, merged_indexes>;
    constexpr bool is_first = (Index == 0);
    constexpr bool is_last = (Index == std::tuple_size<LoopRangesTuple>::value - 1);

    if constexpr (LoopRangeOld::start == NewRange::start && LoopRangeOld::end == NewRange::end) {
        // [-----------]
        // [-----------]
        return loop_with_rhs_merged {};

    } else if constexpr ((LoopRangeOld::end <= NewRange::start)) {
        // [-----------]
        //              [-----------]
        // disjunct
        if constexpr (is_last) {
            // Disjunct but we have to append the new loop part and possibly an
            // empty range to avoid gaps between ranges.
            using new_part
                = basic_looper_union_range<iter, NewRange::start, NewRange::end, typename NewRange::indexes_t>;

            if constexpr (LoopRangeOld::end == NewRange::start) {
                return std::tuple<LoopRangeOld, new_part> {};

            } else {
                using empty_part
                    = basic_looper_union_range<iter, LoopRangeOld::end, NewRange::start, std::make_index_sequence<0>>;

                return std::tuple<LoopRangeOld, empty_part, new_part> {};
            }
        } else {
            return LoopRangeOld {};
        }

    } else if constexpr (LoopRangeOld::start >= NewRange::end) {
        //              [-----------]
        // [-----------]
        // disjunct
        if constexpr (is_first) {
            // Disjunct but we have to append the new loop part and possibly add an
            // empty range to avoid gaps between ranges.
            using new_part
                = basic_looper_union_range<iter, NewRange::start, NewRange::end, typename NewRange::indexes_t>;

            if constexpr (LoopRangeOld::start == NewRange::end) {
                return std::tuple<new_part, LoopRangeOld> {};

            } else {
                using empty_part
                    = basic_looper_union_range<iter, NewRange::end, LoopRangeOld::start, std::make_index_sequence<0>>;
                return std::tuple<new_part, empty_part, LoopRangeOld> {};
            }

        } else {
            return LoopRangeOld {};
        }

    } else if constexpr (LoopRangeOld::start > NewRange::start && LoopRangeOld::end < NewRange::end) {
        // Old part is included in new part
        //    [-----------]
        // [------------------]
        using l = basic_looper_union_range<iter, NewRange::start, LoopRangeOld::start, typename NewRange::indexes_t>;
        using r = basic_looper_union_range<iter, LoopRangeOld::end, NewRange::end, typename NewRange::indexes_t>;

        if constexpr (is_first && is_last) {
            return std::tuple<l, loop_with_rhs_merged, r> {};

        } else if constexpr (is_first) {
            return std::tuple<l, loop_with_rhs_merged> {};

        } else if constexpr (is_last) {
            return std::tuple<loop_with_rhs_merged, r> {};

        } else {
            return loop_with_rhs_merged {};
        }

    } else if constexpr (LoopRangeOld::start == NewRange::start && LoopRangeOld::end < NewRange::end) {
        // Old part is included in new part with same start
        // [-----------]
        // [------------------]
        using l = basic_looper_union_range<iter, LoopRangeOld::start, LoopRangeOld::end, merged_indexes>;

        if constexpr (is_last) {
            using r = basic_looper_union_range<iter, LoopRangeOld::end, NewRange::end, typename NewRange::indexes_t>;
            return std::tuple<l, r> {};

        } else {
            return std::tuple<l> {};
        }

    } else if constexpr (LoopRangeOld::start > NewRange::start && LoopRangeOld::end == NewRange::end) {
        // Old part is included in new part with same end
        //        [-----------]
        // [------------------]
        using r = basic_looper_union_range<iter, LoopRangeOld::start, LoopRangeOld::end, merged_indexes>;

        if constexpr (is_last) {
            using l
                = basic_looper_union_range<iter, NewRange::start, LoopRangeOld::start, typename NewRange::indexes_t>;
            return std::tuple<l, r> {};

        } else {
            return std::tuple<r> {};
        }

    } else if constexpr (LoopRangeOld::start == NewRange::start && LoopRangeOld::end > NewRange::end) {
        // [-----------------]
        // [-----------]
        // start is the same but end is smaller than the original so we have to create two parts
        using l = basic_looper_union_range<iter, LoopRangeOld::start, NewRange::end, merged_indexes>;
        using r = basic_looper_union_range<iter, NewRange::end, LoopRangeOld::end, typename LoopRangeOld::indexes_t>;
        return std::tuple<l, r>();

    } else if constexpr (LoopRangeOld::start < NewRange::start && LoopRangeOld::end == NewRange::end) {
        // [-----------------]
        //      [------------]
        // end is the same but start is higher than the original so we have to create two parts
        using l
            = basic_looper_union_range<iter, LoopRangeOld::start, NewRange::start, typename LoopRangeOld::indexes_t>;
        using r = basic_looper_union_range<iter, NewRange::start, LoopRangeOld::end, merged_indexes>;
        return std::tuple<l, r>();

    } else if constexpr (LoopRangeOld::start < NewRange::start && LoopRangeOld::end > NewRange::end) {
        // [-------------------]
        //     [-----------]
        // the new loop is included but is not the same as the old, so we have to create three parts.
        using l
            = basic_looper_union_range<iter, LoopRangeOld::start, NewRange::start, typename LoopRangeOld::indexes_t>;
        using c = basic_looper_union_range<iter, NewRange::start, NewRange::end, merged_indexes>;
        using r = basic_looper_union_range<iter, NewRange::end, LoopRangeOld::end, typename LoopRangeOld::indexes_t>;

        return std::tuple<l, c, r>();

    } else if constexpr (LoopRangeOld::start < NewRange::start && LoopRangeOld::end < NewRange::end) {
        // [-----------]
        //      [-----------]
        using l
            = basic_looper_union_range<iter, LoopRangeOld::start, NewRange::start, typename LoopRangeOld::indexes_t>;
        using c = basic_looper_union_range<iter, NewRange::start, LoopRangeOld::end, merged_indexes>;
        if constexpr (is_last) {
            // We have to create the last piece as well
            using r = basic_looper_union_range<iter, LoopRangeOld::end, NewRange::end, typename NewRange::indexes_t>;
            return std::tuple<l, c, r>();
        } else {
            return std::tuple<l, c>();
        }
    } else if constexpr (LoopRangeOld::start > NewRange::start && LoopRangeOld::end > NewRange::end) {
        //      [-----------]
        // [-----------]
        using c = basic_looper_union_range<iter, LoopRangeOld::start, NewRange::end, merged_indexes>;
        using r = basic_looper_union_range<iter, NewRange::end, LoopRangeOld::end, typename LoopRangeOld::indexes_t>;

        if constexpr (is_first) {
            using l
                = basic_looper_union_range<iter, NewRange::start, LoopRangeOld::start, typename NewRange::indexes_t>;
            // We have to create the first piece as well
            return std::tuple<l, c, r>();
        } else {
            return std::tuple<c, r>();
        }
    }
}

/// Merges two loop parts.
template <class LoopRangesTuple, class IndexSequence, class NewRange>
struct merge_loop_range_impl;

template <class... LoopRanges, std::size_t... Indexes, class NewRange>
struct merge_loop_range_impl<std::tuple<LoopRanges...>, std::index_sequence<Indexes...>, NewRange> {
    // for each Range in LoopRanges defined by "Indexes", merge the new range
    using type = std::tuple<decltype(merge_with_range_impl<std::tuple<LoopRanges...>, Indexes, NewRange>())...>;
};

template <class LoopRangesTuple, class NewRange>
using merge_loop_range = types::flatten_tuple_t< //
    typename merge_loop_range_impl< //
        LoopRangesTuple, //
        std::make_index_sequence<std::tuple_size_v<LoopRangesTuple>>, //
        NewRange>::type>;

template <class LooperUnion, class Looper>
constexpr auto add_to_union(LooperUnion lhs, Looper rhs) {
    if constexpr (!LooperUnion::template is_loop_function<Looper>::value) {
        static_assert(
            std::is_same_v<typename LooperUnion::iterator, typename Looper::iterator>, "Iterators must be the same");

        // Single loop part for the rhs.
        using new_loop_indexes = types::index_sequence_from_to< //
            LooperUnion::function_count, //
            LooperUnion::function_count + Looper::function_count>;

        using rhs_part = basic_looper_union_range<std::size_t, Looper::start, Looper::end, new_loop_indexes>;

        auto merged_functions = std::tuple_cat(lhs.functions, rhs.functions);

        using merged_loop_ranges = merge_loop_range<typename LooperUnion::loops, rhs_part>;
        using merged_types = std::pair<merged_loop_ranges, decltype(merged_functions)>;

        return basic_looper_union<typename Looper::iterator, merged_types>(std::move(merged_functions));
    } else {
        // return type void => crashes if assigned to variable or .run() method is invoked
    }
}

} // namespace loop_fusion::compiletime
