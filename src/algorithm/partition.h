#pragma once 

#include <algorithm>
#include <iterator>
#include <utility>

#include "algorithm/rotate.h"


/// TODO doc 
/// TODO parallel versions 
/// TODO simd versions

template<class InputIt,
         class UnaryPred>
constexpr 
bool is_partitioned(InputIt first,
                    InputIt last,
                    UnaryPred p) {
    for (; first != last; ++first)
        if (!p(*first))
            break;
    for (; first != last; ++first)
        if (p(*first))
            return false;
    return true;
}

template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt partition_point(ForwardIt first,
                          ForwardIt last,
                          UnaryPred p) {
    for (auto length = std::distance(first, last); 0 < length; )
    {
        auto half = length / 2;
        auto middle = std::next(first, half);
        if (p(*middle))
        {
            first = std::next(middle);
            length -= (half + 1);
        }
        else
            length = half;
    }
 
    return first;
}

template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt partition(ForwardIt first,
                    ForwardIt last, 
                    UnaryPred p) {
    first = std::find_if_not(first, last, p);
    if (first == last)
        return first;
 
    for (auto i = std::next(first); i != last; ++i)
        if (p(*i))
        {
            std::iter_swap(i, first);
            ++first;
        }
 
    return first;
}

template<class InputIt,
         class OutputIt1,
         class OutputIt2,
         class UnaryPred>
constexpr
std::pair<OutputIt1, OutputIt2>
    partition_copy(InputIt first,
                   InputIt last,
                   OutputIt1 d_first_true,
                   OutputIt2 d_first_false,
                   UnaryPred p) {
    for (; first != last; ++first) {
        if (p(*first)) {
            *d_first_true = *first;
            ++d_first_true;
        } else {
            *d_first_false = *first;
            ++d_first_false;
        }
    }
 
    return std::pair<OutputIt1, OutputIt2>(d_first_true, d_first_false);
}

namespace cryptanalysislib::internal {
    /// stable partition of the `len` elements starting at `first`
    /// divide and conquer: partition both halves, then rotate the
    /// `false` part of the left half behind the `true` part of the right half.
    /// \return iterator to the first element for which `p` is false
    template<class ForwardIt,
             class UnaryPred>
    constexpr ForwardIt stable_partition_rec(ForwardIt first,
                                             const size_t len,
                                             UnaryPred &p) {
        if (len == 1) {
            ForwardIt next = first;
            ++next;
            return p(*first) ? next : first;
        }

        const size_t half = len / 2;
        ForwardIt middle = first;
        for (size_t i = 0; i < half; ++i) { ++middle; }

        ForwardIt left = stable_partition_rec(first, half, p);
        ForwardIt right = stable_partition_rec(middle, len - half, p);
        // [left, middle) are false, [middle, right) are true
        return cryptanalysislib::rotate(left, middle, right);
    }
} // end namespace cryptanalysislib::internal

/// Reorders [first, last) such that all elements for which `p` is true come
/// first, keeping the relative order within both groups.
/// NOTE: in place; `p` is called exactly once per element; O(n log n) moves.
/// \return iterator to the first element of the second group
template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt stable_partition(ForwardIt first,
                           ForwardIt last, 
                           UnaryPred p) {
    // skip the prefix, which is already in place
    while ((first != last) && p(*first)) {
        ++first;
    }

    if (first == last) {
        return first;
    }

    // `*first` is known to be false: partition the rest and move `*first`
    // behind its `true` elements
    ForwardIt next = first;
    ++next;

    size_t len = 0;
    for (ForwardIt it = next; it != last; ++it) {
        len += 1;
    }

    const ForwardIt r = (len == 0) ? next : cryptanalysislib::internal::stable_partition_rec(next, len, p);
    return cryptanalysislib::rotate(first, next, r);
}
