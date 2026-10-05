#pragma once 

#include <pair>


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

template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt stable_partition(ForwardIt first,
                           ForwardIt last, 
                           UnaryPred p) {
    // TODO
}
