#pragma once 

#include "copy.h"

// TODO parallel versions
// TODO simd versions

namespace cryptanalysislib {

    /// TODO doc
    template<class InputIt1,
             class InputIt2>
    bool includes(InputIt1 first1,
                  InputIt1 last1,
                  InputIt2 first2,
                  InputIt2 last2) {
        for (; first2 != last2; ++first1) {
            if (first1 == last1 || *first2 < *first1)
                return false;
            if (!(*first1 < *first2))
                ++first2;
        }
    
        return true;
    }

    /// TODO doc
    template<class InputIt1,
             class InputIt2,
             class Compare>
    bool includes(InputIt1 first1,
                  InputIt1 last1,
                  InputIt2 first2,
                  InputIt2 last2,
                  Compare comp) {
        for (; first2 != last2; ++first1) {
            if (first1 == last1 || comp(*first2, *first1))
                return false;
            if (!comp(*first1, *first2))
                ++first2;
        }

        return true;
    }

template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_difference(InputIt1 first1, InputIt1 last1,
                        InputIt2 first2, InputIt2 last2, OutputIt d_first)
{
    while (first1 != last1)
    {
        if (first2 == last2)
            return std::copy(first1, last1, d_first);
 
        if (*first1 < *first2)
            *d_first++ = *first1++;
        else
        {
            if (! (*first2 < *first1))
                ++first1;
            ++first2;
        }
    }
    return d_first;
}

template<class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt set_difference(InputIt1 first1, InputIt1 last1,
                        InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp)
{
    while (first1 != last1)
    {
        if (first2 == last2)
            return std::copy(first1, last1, d_first);
 
        if (comp(*first1, *first2))
            *d_first++ = *first1++;
        else
        {
            if (!comp(*first2, *first1))
                ++first1;
            ++first2;
        }
    }
    return d_first;
}

template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_intersection(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2, InputIt2 last2, OutputIt d_first)
{
    while (first1 != last1 && first2 != last2)
    {
        if (*first1 < *first2)
            ++first1;
        else
        {
            if (!(*first2 < *first1))
                *d_first++ = *first1++; // *first1 and *first2 are equivalent.
            ++first2;
        }
    }
    return d_first;
}
template<class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt set_intersection(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp)
{
    while (first1 != last1 && first2 != last2)
    {
        if (comp(*first1, *first2))
            ++first1;
        else
        {
            if (!comp(*first2, *first1))
                *d_first++ = *first1++; // *first1 and *first2 are equivalent.
            ++first2;
        }
    }
    return d_first;
}

template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_symmetric_difference(InputIt1 first1, InputIt1 last1,
                                  InputIt2 first2, InputIt2 last2, OutputIt d_first)
{
    while (first1 != last1)
    {
        if (first2 == last2)
            return std::copy(first1, last1, d_first);
 
        if (*first1 < *first2)
            *d_first++ = *first1++;
        else
        {
            if (*first2 < *first1)
                *d_first++ = *first2;
            else
                ++first1;
            ++first2;
        }
    }
    return std::copy(first2, last2, d_first);
}

template<class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt set_symmetric_difference(InputIt1 first1, InputIt1 last1,
                                  InputIt2 first2, InputIt2 last2,
                                  OutputIt d_first, Compare comp)
{
    while (first1 != last1)
    {
        if (first2 == last2)
            return std::copy(first1, last1, d_first);
 
        if (comp(*first1, *first2))
            *d_first++ = *first1++;
        else
        {
            if (comp(*first2, *first1))
                *d_first++ = *first2;
            else
                ++first1;
            ++first2;
        }
    }
    return std::copy(first2, last2, d_first);
}
template<class InputIt1, class InputIt2, class OutputIt>
OutputIt set_union(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2, OutputIt d_first)
{
    for (; first1 != last1; ++d_first)
    {
        if (first2 == last2)
            return std::copy(first1, last1, d_first);
 
        if (*first2 < *first1)
            *d_first = *first2++;
        else
        {
            *d_first = *first1;
            if (!(*first1 < *first2))
                ++first2;
            ++first1;
        }
    }
    return std::copy(first2, last2, d_first);
}
template<class InputIt1, class InputIt2, class OutputIt, class Compare>
OutputIt set_union(InputIt1 first1, InputIt1 last1,
                   InputIt2 first2, InputIt2 last2, OutputIt d_first, Compare comp)
{
    for (; first1 != last1; ++d_first)
    {
        if (first2 == last2)
            // Finished range 2, include the rest of range 1:
            return std::copy(first1, last1, d_first);
 
        if (comp(*first2, *first1))
            *d_first = *first2++;
        else
        {
            *d_first = *first1;
            if (!comp(*first1, *first2)) // Equivalent => don't need to include *first2.
                ++first2;
            ++first1;
        }
    }
    // Finished range 1, include the rest of range 2:
    return std::copy(first2, last2, d_first);
}
}; // end namespace cryptanalysislib
