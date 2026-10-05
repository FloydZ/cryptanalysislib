#pragma once 

#include "copy.h"
#include "simd/simd.h"

// TODO parallel versions

namespace cryptanalysislib {
    namespace internal {
        /// TODO tests
        /// SIMD-optimized set_difference for unsigned integer types
        /// Computes the set difference of two sorted arrays: elements from the first array that are not in the second
        /// \tparam T Unsigned integer type
        /// \param data1[in]: Pointer to first sorted array
        /// \param n1[in]: Number of elements in the first array
        /// \param data2[in]: Pointer to second sorted array
        /// \param n2[in]: Number of elements in the second array
        /// \param dest[out]: Pointer to destination array
        /// \return Number of elements in the result
        template<typename T>
#if __cplusplus > 201709L
        requires std::unsigned_integral<T>
#endif
        size_t set_difference_uXX_simd(const T* data1, const size_t n1,
                                       const T* data2, const size_t n2,
                                       T* dest) noexcept {
            if (n1 == 0) {
                return 0; // First set is empty, result is empty
            }
            if (n2 == 0) {
                // Second set is empty, copy all elements from first set
                std::copy(data1, data1 + n1, dest);
                return n1;
            }
            
            using S = SIMDSelector<T>;
            
            size_t i1 = 0, i2 = 0, dest_idx = 0;
            
            while (i1 < n1) {
                // If we reached the end of data2, copy remaining elements from data1
                if (i2 == n2) {
                    std::copy(data1 + i1, data1 + n1, dest + dest_idx);
                    dest_idx += (n1 - i1);
                    break;
                }
                
                // If we have enough elements left in data1 for a SIMD comparison
                if (i1 + S::LIMBS <= n1) {
                    // Load a SIMD-width chunk of data1
                    const auto chunk1 = S::load<false>(data1 + i1);
                    // Create a SIMD vector with the current data2 element replicated
                    const auto val2 = S::set1(data2[i2]);
                    
                    // Compare if all elements in chunk1 are less than data2[i2]
                    const auto less_mask = chunk1 < val2;
                    
                    if (less_mask == S::maskFull()) {
                        // All elements in chunk1 are less than data2[i2], 
                        // so they all belong in the difference
                        S::store<false>(dest + dest_idx, chunk1);
                        dest_idx += S::LIMBS;
                        i1 += S::LIMBS;
                    } else {
                        // Process chunk1 elements one by one
                        for (size_t j = 0; j < S::LIMBS && i1 < n1; ++j) {
                            if (data1[i1] < data2[i2]) {
                                // Element in data1 is less than current data2 element,
                                // so it belongs in the difference
                                dest[dest_idx++] = data1[i1++];
                            } else if (data1[i1] == data2[i2]) {
                                // Elements are equal, skip the element from data1
                                ++i1;
                                ++i2;
                                // If we reached the end of data2, break to handle rest of data1
                                if (i2 == n2) break;
                            } else {
                                // Element in data1 is greater than current data2 element,
                                // advance data2 pointer
                                ++i2;
                                // If we reached the end of data2, break to handle rest of data1
                                if (i2 == n2) break;
                                // Need to recheck current data1 element with next data2 element
                                --j;
                            }
                        }
                    }
                } else {
                    // Not enough elements left for SIMD, fall back to scalar
                    if (data1[i1] < data2[i2]) {
                        // Element in data1 is less than current data2 element,
                        // so it belongs in the difference
                        dest[dest_idx++] = data1[i1++];
                    } else if (data1[i1] == data2[i2]) {
                        // Elements are equal, skip the element from data1
                        ++i1;
                        ++i2;
                    } else {
                        // Element in data1 is greater than current data2 element,
                        // advance data2 pointer
                        ++i2;
                    }
                }
            }
            
            return dest_idx;
        }

        /// TODO tests
        /// SIMD-optimized includes for unsigned integer types
        /// Checks if the sorted range [first2, last2) is a subset of the sorted range [first1, last1)
        /// \tparam T Unsigned integer type
        /// \param data1[in]: Pointer to first sorted array
        /// \param n1[in]: Number of elements in the first array
        /// \param data2[in]: Pointer to second sorted array
        /// \param n2[in]: Number of elements in the second array
        /// \return true if second array is subset of first, false otherwise
        template<typename T>
#if __cplusplus > 201709L
        requires std::unsigned_integral<T>
#endif
        bool includes_uXX_simd(const T* data1,
                               const size_t n1,
                               const T* data2, 
                               const size_t n2) noexcept {
            if (n2 == 0) {
                return true; // Empty set is always a subset
            }
            if (n1 == 0) {
                return false; // Non-empty set cannot be a subset of empty set
            }
            
            using S = SIMDSelector<T>;
            
            size_t i1 = 0, i2 = 0;
            
            // Process elements with SIMD when possible
            while (i1 < n1 && i2 < n2) {
                // If we have enough elements left in data1 for a SIMD comparison
                if (i1 + S::LIMBS <= n1) {
                    // Load a SIMD-width chunk of data1
                    const auto chunk1 = S::load<false>(data1 + i1);
                    // Create a SIMD vector with the current data2 element replicated
                    const auto val2 = S::set1(data2[i2]);
                    
                    // Compare if any element in chunk1 equals val2
                    const auto mask = chunk1 == val2;
                    
                    if (mask) {
                        // Found a match, advance to next data2 element
                        ++i2;
                        // Advance i1 to first element after the match
                        const size_t match_pos = ffs<T>(mask) - 1u;
                        i1 += match_pos + 1;
                    } else {
                        // No match in this chunk
                        // Check if all elements in chunk1 are less than data2[i2]
                        const auto less_mask = chunk1 < val2;
                        
                        if (less_mask == S::maskFull()) {
                            // All elements are less, advance to next chunk
                            i1 += S::LIMBS;
                        } else {
                            // Some elements are greater, which means data2[i2] is not in data1
                            return false;
                        }
                    }
                } else {
                    // Not enough elements left for SIMD, fall back to scalar
                    if (data1[i1] < data2[i2]) {
                        ++i1;
                    } else if (data2[i2] < data1[i1]) {
                        return false; // Element in data2 not found in data1
                    } else { // Equal elements
                        ++i1;
                        ++i2;
                    }
                }
            }
            // todo definelty not correct
            
            // If we've processed all elements in data2, it's a subset
            return i2 == n2;
        }
    }
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
        using T = typename std::iterator_traits<InputIt1>::value_type;
        
        // For contiguous iterators and unsigned integer types, use SIMD version
        if constexpr (std::is_unsigned_v<T> && 
                      std::is_same_v<typename std::iterator_traits<InputIt1>::iterator_category, 
                                    std::random_access_iterator_tag> &&
                      std::is_same_v<typename std::iterator_traits<InputIt2>::iterator_category, 
                                    std::random_access_iterator_tag>) {
            return internal::includes_uXX_simd<T>(&(*first1), 
                                   static_cast<size_t>(std::distance(first1, last1)),
                                   &(*first2),
                                   static_cast<size_t>(std::distance(first2, last2)));
        }
        
        // Standard implementation for non-SIMD compatible types
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
        using T = typename std::iterator_traits<InputIt1>::value_type;
        
        // For contiguous iterators and unsigned integer types, use SIMD version
        if constexpr (std::is_unsigned_v<T> && 
                      std::is_same_v<typename std::iterator_traits<InputIt1>::iterator_category, 
                                    std::random_access_iterator_tag> &&
                      std::is_same_v<typename std::iterator_traits<InputIt2>::iterator_category, 
                                    std::random_access_iterator_tag> &&
                      std::is_same_v<typename std::iterator_traits<OutputIt>::iterator_category, 
                                    std::random_access_iterator_tag>) {
            const size_t n1 = static_cast<size_t>(std::distance(first1, last1));
            const size_t n2 = static_cast<size_t>(std::distance(first2, last2));
            
            // Create temporary buffer for results
            std::vector<T> temp_buffer(n1);
            
            // Call SIMD implementation
            const size_t result_size = internal::set_difference_uXX_simd<T>(
                &(*first1), n1, 
                &(*first2), n2,
                temp_buffer.data()
            );
            
            // Copy result to output iterator
            std::copy(temp_buffer.data(), temp_buffer.data() + result_size, d_first);
            std::advance(d_first, result_size);
            return d_first;
        }
        
        // Standard implementation for non-SIMD compatible types
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
