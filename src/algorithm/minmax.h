#pragma once

#include <utility>
#include <cstdlib>
#include <concepts>

#include "simd/simd.h"

/// TODO parallel version implementation

namespace cryptanalysislib {
    
    namespace internal {

        /// TODO make iterator version of this
        /// SIMD implementation for minmax_element that finds min and max values in a range
        /// Uses SIMD operations for improved performance with larger data sets
        /// \param first[in]: pointer to the first element in the array
        /// \param last[in]: pointer to one past the last element in the array
        /// \return a pair containing the minimum and maximum values in the range
        template<class T>
        constexpr std::pair<T, T> simd_minmax_element(const T* first, 
                                                      const T* last) noexcept {
            using S = SIMDSelector<T>;
            constexpr size_t SIMD_WIDTH = S::LIMBS;
            
            // Handle empty array
            if (first == last) {
                return {T{}, T{}};
            }
            
            const size_t size = last - first;
            
            // If size too small for SIMD or no SIMD available, use scalar version
            if (size < SIMD_WIDTH || SIMD_WIDTH == 1) {
                T min_val = *first;
                T max_val = *first;
                
                for (auto it = first + 1; it != last; ++it) {
                    const T val = *it;
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                
                return {min_val, max_val};
            }
            
            // Initialize min/max with first SIMD_WIDTH elements
            S vec = S::loadu(first);
            S min_vec = vec;
            S max_vec = vec;
            
            // Process data in chunks of SIMD_WIDTH
            const size_t simd_chunks = (size / SIMD_WIDTH);
            for (size_t i = 1; i < simd_chunks; ++i) {
                vec = S::loadu(first + i * SIMD_WIDTH);
                min_vec = S::min(min_vec, vec);
                max_vec = S::max(max_vec, vec);
            }
            
            // Extract min/max values from SIMD vectors
            T min_arr[SIMD_WIDTH];
            T max_arr[SIMD_WIDTH];
            S::storeu(min_arr, min_vec);
            S::storeu(max_arr, max_vec);
            
            T min_val = min_arr[0];
            T max_val = max_arr[0];
            
            // Find min/max within the SIMD results
            for (size_t i = 1; i < SIMD_WIDTH; ++i) {
                if (min_arr[i] < min_val) min_val = min_arr[i];
                if (max_arr[i] > max_val) max_val = max_arr[i];
            }
            
            // Process remaining elements
            for (size_t i = simd_chunks * SIMD_WIDTH; i < size; ++i) {
                const T val = first[i];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            
            return {min_val, max_val};
        }
        
        /// Wrapper for simd_minmax_element that works with iterators
        /// \param first[in]: iterator to the first element
        /// \param last[in]: iterator to one past the last element
        /// \return a pair containing the minimum and maximum values in the range
        template<class ForwardIt>
        std::pair<typename std::iterator_traits<ForwardIt>::value_type,
                  typename std::iterator_traits<ForwardIt>::value_type> 
        minmax_element_simd(ForwardIt first,
                            ForwardIt last) noexcept {
            using T = typename std::iterator_traits<ForwardIt>::value_type;
            static_assert(std::unsigned_integral<T>);
            
            // Check if we can use pointers for faster access
            if constexpr (std::contiguous_iterator<ForwardIt>) {
                return simd_minmax_element(&(*first), &(*last));
            } else {
                // Fallback for non-contiguous iterators
                if (first == last) {
                    return {T{}, T{}};
                }
                
                T min_val = *first;
                T max_val = *first;
                
                for (auto it = std::next(first); it != last; ++it) {
                    const T val = *it;
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                
                return {min_val, max_val};
            }
        }
    }; // end namespace internal

    /// Computes the minimum and maximum of two values
    /// Returns a pair consisting of the smaller value followed by the larger value
    /// \param a[in]: first value to compare
    /// \param b[in]: second value to compare
    /// \return a pair containing references to the minimum and maximum values
    template<class T>
    constexpr 
    std::pair<const T&, const T&> minmax(const T& a,
                                         const T& b) noexcept {
        return (b < a) ? std::pair<const T&, const T&>(b, a)
                       : std::pair<const T&, const T&>(a, b);
    }

    /// Computes the minimum and maximum of two values using a custom comparator
    /// Returns a pair consisting of the smaller value followed by the larger value
    /// \param a[in]: first value to compare
    /// \param b[in]: second value to compare
    /// \param comp[in]: binary comparison function object
    /// \return a pair containing references to the minimum and maximum values
    template<class T, class Compare>
    constexpr 
    std::pair<const T&, const T&> minmax(const T& a,
                                         const T& b,
                                         Compare comp) noexcept {
        return comp(b, a) ? std::pair<const T&, const T&>(b, a)
                          : std::pair<const T&, const T&>(a, b);
    }
    
    /// Computes the minimum and maximum elements in an initializer list
    /// \param ilist[in]: initializer list of values to examine
    /// \return a pair containing copies of the minimum and maximum values
    template<class T>
    constexpr 
    std::pair<T, T> minmax(std::initializer_list<T> ilist) noexcept {
        auto p = minmax_element(ilist.begin(), ilist.end());
        return std::pair(*p.first, *p.second);
    }

    /// Finds the smallest and largest elements in a range
    /// Uses the less-than operator for comparison
    /// \param first[in]: iterator to the first element in the range
    /// \param last[in]: iterator to one past the last element in the range
    /// \return a pair of iterators pointing to the minimum and maximum elements
    template<class ForwardIt>
    constexpr 
    std::pair<ForwardIt, ForwardIt>
        minmax_element(ForwardIt first,
                       ForwardIt last) {
        using value_type = typename std::iterator_traits<ForwardIt>::value_type;
        return minmax_element(first, last, std::less<value_type>());
    }
    
    /// Finds the smallest and largest elements in a range using a custom comparator
    /// The function processes elements in pairs to minimize the total number of comparisons
    /// \param first[in]: iterator to the first element in the range
    /// \param last[in]: iterator to one past the last element in the range
    /// \param comp[in]: binary comparison function object that defines element ordering
    /// \return a pair of iterators pointing to the minimum and maximum elements
    template<class ForwardIt,
             class Compare>
    constexpr 
    std::pair<ForwardIt, ForwardIt>
        minmax_element(ForwardIt first,
                       ForwardIt last,
                       Compare comp) {
        auto min = first, max = first;
     
        if (first == last || ++first == last)
            return {min, max};
     
        if (comp(*first, *min))
            min = first;
        else
            max = first;
     
        while (++first != last)
        {
            auto i = first;
            if (++first == last)
            {
                if (comp(*i, *min))
                    min = i;
                else if (!(comp(*i, *max)))
                    max = i;
                break;
            }
            else
            {
                if (comp(*first, *i))
                {
                    if (comp(*first, *min))
                        min = first;
                    if (!(comp(*i, *max)))
                        max = i;
                }
                else
                {
                    if (comp(*i, *min))
                        min = i;
                    if (!(comp(*first, *max)))
                        max = first;
                }
            }
        }
        return {min, max};
    }

}; // end namespace cryptanalysislib
