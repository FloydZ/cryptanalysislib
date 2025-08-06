#pragma once 

#include "copy.h"
#include "simd/simd.h"

// TODO parallel versions
// TODO simd versions

namespace cryptanalysislib {
    namespace internal {
        template<class InputIt1,
                 class InputIt2>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt1> &&
                 std::forward_iterator<InputIt2>
#endif
        bool includes(InputIt1 first1,
                      InputIt1 last1,
                      InputIt2 first2,
                      InputIt2 last2) {

		    using T = InputIt1::value_type;
            using S = SIMDSelector<T>;
            constexpr size_t SIMD_WIDTH = S::LIMBS;
        }
    };

    /// Checks if the sorted range [first2, last2) is a subset of the sorted range [first1, last1)
    /// Both input ranges must be sorted in the same order
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \return true if the second range is a subset of the first, false otherwise
    template<class InputIt1,
             class InputIt2>
    bool includes(InputIt1 first1,
                  InputIt1 last1,
                  InputIt2 first2,
                  InputIt2 last2) {
        for (; first2 != last2; ++first1) {
            if (first1 == last1 || *first2 < *first1) {
                return false;
            }
            if (!(*first1 < *first2)) {
                ++first2;
            }
        }
    
        return true;
    }

    /// Checks if the sorted range [first2, last2) is a subset of the sorted range [first1, last1)
    /// Both input ranges must be sorted according to the comparison function
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param comp[in]: binary comparison function object that defines the element ordering
    /// \return true if the second range is a subset of the first, false otherwise
    template<class InputIt1,
             class InputIt2,
             class Compare>
    bool includes(InputIt1 first1,
                  InputIt1 last1,
                  InputIt2 first2,
                  InputIt2 last2,
                  Compare comp) {
        for (; first2 != last2; ++first1) {
            if (first1 == last1 || comp(*first2, *first1)) {
                return false;
            }
            if (!comp(*first1, *first2)) {
                ++first2;
            }
        }

        return true;
    }

    /// Computes the difference of two sorted ranges: elements from the first range that are not in the second
    /// Both input ranges must be sorted in the same order
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \return iterator to the end of the constructed range
    template<class InputIt1, class InputIt2, class OutputIt>
    OutputIt set_difference(InputIt1 first1,
                            InputIt1 last1,
                            InputIt2 first2,
                            InputIt2 last2,
                            OutputIt d_first) {
        while (first1 != last1) {
            if (first2 == last2) {
                return std::copy(first1, last1, d_first);
            }

            if (*first1 < *first2) {
                *d_first++ = *first1++;
            } else {
                if (! (*first2 < *first1))
                    ++first1;
                ++first2;
            }
        }
        return d_first;
    }
    
    /// Computes the difference of two sorted ranges: elements from the first range that are not in the second
    /// Both input ranges must be sorted according to the comparison function
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: binary comparison function object that defines the element ordering
    /// \return iterator to the end of the constructed range
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
    
    /// Computes the intersection of two sorted ranges: elements that are present in both ranges
    /// Both input ranges must be sorted in the same order
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \return iterator to the end of the constructed range
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
    
    /// Computes the intersection of two sorted ranges: elements that are present in both ranges
    /// Both input ranges must be sorted according to the comparison function
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: binary comparison function object that defines the element ordering
    /// \return iterator to the end of the constructed range
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
    
    /// Computes the symmetric difference of two sorted ranges: elements in either range but not in both
    /// Both input ranges must be sorted in the same order
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \return iterator to the end of the constructed range
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
    
    /// Computes the symmetric difference of two sorted ranges: elements in either range but not in both
    /// Both input ranges must be sorted according to the comparison function
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: binary comparison function object that defines the element ordering
    /// \return iterator to the end of the constructed range
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
    /// Computes the union of two sorted ranges: elements that are present in either or both ranges
    /// Both input ranges must be sorted in the same order
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \return iterator to the end of the constructed range
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
    
    /// Computes the union of two sorted ranges: elements that are present in either or both ranges
    /// Both input ranges must be sorted according to the comparison function
    /// \param first1[in]: iterator to the first element of the first range
    /// \param last1[in]: iterator to one past the last element of the first range
    /// \param first2[in]: iterator to the first element of the second range
    /// \param last2[in]: iterator to one past the last element of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: binary comparison function object that defines the element ordering
    /// \return iterator to the end of the constructed range
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
