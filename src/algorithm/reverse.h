#pragma once 
#include <iterator>


/// TODO simd and parallel version

/// TODO doc
template<class BidirIt>
constexpr
void reverse(BidirIt first,
             BidirIt last) noexcept {
    using iter_cat = typename std::iterator_traits<BidirIt>::iterator_category;
 
    // Tag dispatch, e.g. calling reverse_impl(first, last, iter_cat()),
    // can be used in C++14 and earlier modes.
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iter_cat>) {
        if (first == last) {
            return;
        }
 
        for (--last; first < last; (void)++first, --last) {
            std::iter_swap(first, last);
        }
    } else {
        while (first != last && first != --last) {
            std::iter_swap(first++, last);
        }
    }
}

/// TODO doc
template<class BidirIt,
         class OutputIt>
constexpr 
OutputIt reverse_copy(BidirIt first,
                      BidirIt last,
                      OutputIt d_first) {
    for (; first != last; ++d_first) {
        *d_first = *(--last);
    }
    return d_first;
}
