#pragma once 

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>

#include "algorithm/swap.h"
#include "sort/ska_sort.h"

namespace cryptanalysislib {
    template<class ForwardIt, class Compare>
    constexpr ForwardIt is_sorted_until(ForwardIt first, ForwardIt last, Compare comp);

    /// Finds the end of the longest sorted prefix of [first, last)
    /// (like `std::is_sorted_until`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \return first iterator `it` with `*it < *(it - 1)`, or last
    template<class ForwardIt>
    constexpr //< since C++20
    ForwardIt is_sorted_until(ForwardIt first,
                              ForwardIt last) {
        return cryptanalysislib::is_sorted_until(first, last, std::less<>());
    }

    /// Finds the end of the longest prefix of [first, last) sorted w.r.t.
    /// `comp` (like `std::is_sorted_until`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param comp[in]: less-than comparison
    /// \return first iterator `it` with `comp(*it, *(it - 1))`, or last
    template<class ForwardIt,
             class Compare>
    constexpr
    ForwardIt is_sorted_until(ForwardIt first,
                              ForwardIt last,
                              Compare comp) {
        if (first != last) {
            ForwardIt next = first;
            while (++next != last) {
                if (comp(*next, *first)) {
                    return next;
                }
                first = next;
            }
        }

        return last;
    }
    
    /// Checks whether [first, last) is sorted in non-descending order
    /// (like `std::is_sorted`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \return true if sorted (also if the range is empty)
    template<class ForwardIt>
    bool is_sorted(ForwardIt first,
                   ForwardIt last) {
        return cryptanalysislib::is_sorted_until(first, last) == last;
    }
   
    /// Checks whether [first, last) is sorted w.r.t. `comp` (like `std::is_sorted`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param comp[in]: less-than comparison
    /// \return true if sorted (also if the range is empty)
    template<class ForwardIt, class Compare>
    bool is_sorted(ForwardIt first,
                   ForwardIt last,
                   Compare comp) {
        return cryptanalysislib::is_sorted_until(first, last, comp) == last;
    }

    /// Sorts [first, last) by the key `extract_key(*it)` (radix sort, `ska_sort`).
    /// NOTE: `extract_key` takes a single element, the overload with a
    ///     two argument comparator is in `sort/sort.h`.
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param extract_key[in]: returns the sort key (integer, float, ...) of an element
    template<class RandIt,
             class ExtractKey>
        requires std::random_access_iterator<RandIt> &&
                 std::invocable<ExtractKey &, std::iter_reference_t<RandIt>>
    constexpr inline void sort(RandIt first,
                               RandIt last,
                               ExtractKey extract_key) noexcept {
        ::ska_sort(first, last, extract_key);
    }




template<typename RandomIt, typename Compare>
constexpr void partial_sort(RandomIt first, RandomIt middle, RandomIt last, Compare comp);

/// Rearranges [first, last) such that [first, middle) contains the
/// `middle - first` smallest elements in ascending order (like
/// `std::partial_sort`); the order of the rest is unspecified.
/// NOTE: before, this forwarded to `std::partial_sort`
template<typename RandomIt>
constexpr //< since C++20
void partial_sort(RandomIt first, RandomIt middle, RandomIt last)
{
    typedef typename std::iterator_traits<RandomIt>::value_type VT;
    cryptanalysislib::partial_sort(first, middle, last, std::less<VT>());
}
namespace impl
{
    template<typename RandomIt, typename Compare>
    constexpr //< since C++20
    void sift_down(RandomIt first, RandomIt last, const Compare& comp)
    {
        // sift down element at “first”
        const auto length = static_cast<std::size_t>(last - first);
        std::size_t current = 0;
        std::size_t next = 2;
        while (next < length)
        {
            if (comp(*(first + next), *(first + (next - 1))))
                --next;
            if (!comp(*(first + current), *(first + next)))
                return;
            cryptanalysislib::iter_swap(first + current, first + next);
            current = next;
            next = 2 * current + 2;
        }
        --next;
        if (next < length && comp(*(first + current), *(first + next)))
            cryptanalysislib::iter_swap(first + current, first + next);
    }
 
    template<typename RandomIt, typename Compare>
    constexpr //< since C++20
    void heap_select(RandomIt first, RandomIt middle, RandomIt last, const Compare& comp)
    {
        std::make_heap(first, middle, comp);
        for (auto i = middle; i != last; ++i)
        {
            if (comp(*i, *first))
            {
                cryptanalysislib::iter_swap(first, i);
                impl::sift_down(first, middle, comp);
            }
        }
    }
} // namespace impl
 
/// Rearranges [first, last) such that [first, middle) contains the
/// `middle - first` smallest elements w.r.t. `comp`, sorted (like
/// `std::partial_sort`): heap select + heap sort, O(n log k).
template<typename RandomIt, typename Compare>
constexpr //< since C++20
void partial_sort(RandomIt first, RandomIt middle, RandomIt last, Compare comp)
{
    impl::heap_select(first, middle, last, comp);
    std::sort_heap(first, middle, comp);
}


};
