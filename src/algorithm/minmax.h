#pragma once

#include <utility>
#include <cstdlib>
#include <concepts>

#include "simd/simd.h"
#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    /// Configuration for minmax algorithms
    struct AlgorithmMinMaxConfig : public AlgorithmConfig {
        constexpr static size_t min_size_per_thread = 1u << 15u;
    };
    constexpr static AlgorithmMinMaxConfig algorithmMinMaxConfig;
    
    namespace internal {

        /// NOTE: the iterator version is the overload below
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
            S vec = S::template load<false>(first);
            S min_vec = vec;
            S max_vec = vec;
            
            // Process data in chunks of SIMD_WIDTH
            const size_t simd_chunks = (size / SIMD_WIDTH);
            for (size_t i = 1; i < simd_chunks; ++i) {
                vec = S::template load<false>(first + i * SIMD_WIDTH);
                min_vec = S::min(min_vec, vec);
                max_vec = S::max(max_vec, vec);
            }
            
            // Find min/max within the SIMD results
            T min_val = min_vec[0];
            T max_val = max_vec[0];
            for (size_t i = 1; i < SIMD_WIDTH; ++i) {
                if (min_vec[i] < min_val) min_val = min_vec[i];
                if (max_vec[i] > max_val) max_val = max_vec[i];
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
                if (first == last) {
                    return {T{}, T{}};
                }
                const T *p = &(*first);
                return simd_minmax_element(p, p + (last - first));
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

    /// Finds the smallest and largest elements (parallel version): every chunk
    /// computes its `minmax_element`, which are combined in order. Like the
    /// sequential version (and `std::minmax_element`) this returns the first
    /// smallest and the last largest element.
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first[in]: iterator to the first element in the range
    /// \param last[in]: iterator to one past the last element in the range
    /// \param comp[in]: binary comparison function object
    /// \return a pair of iterators pointing to the minimum and maximum elements
    template<class ExecPolicy,
             class RandIt,
             class Compare,
             const AlgorithmMinMaxConfig &config = algorithmMinMaxConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    std::pair<RandIt, RandIt>
        minmax_element(ExecPolicy &&policy,
                       RandIt first,
                       RandIt last,
                       Compare comp) noexcept {
        const auto size = static_cast<size_t>(std::distance(first, last));
        const uint32_t nthreads = should_par(policy, config, size);
        if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
            return cryptanalysislib::minmax_element(first, last, comp);
        }

        using R = std::pair<RandIt, RandIt>;
        auto futures = internal::parallel_chunk_for_1(
            std::forward<ExecPolicy>(policy),
            first, last,
            [comp](RandIt chunk_first, RandIt chunk_last) {
                return cryptanalysislib::minmax_element(chunk_first, chunk_last, comp);
            },
            static_cast<R *>(nullptr),
            1, nthreads);

        // chunks are in order: the first minimum and the last maximum win
        R best = futures[0].get();
        for (size_t i = 1; i < futures.size(); i++) {
            const R c = futures[i].get();
            if (comp(*c.first, *best.first)) { best.first = c.first; }
            if (!comp(*c.second, *best.second)) { best.second = c.second; }
        }
        return best;
    }

    /// parallel `minmax_element` with `<`
    template<class ExecPolicy,
             class RandIt,
             const AlgorithmMinMaxConfig &config = algorithmMinMaxConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    std::pair<RandIt, RandIt>
        minmax_element(ExecPolicy &&policy,
                       RandIt first,
                       RandIt last) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;
        return cryptanalysislib::minmax_element(std::forward<ExecPolicy>(policy),
                                                first, last, std::less<T>());
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

}; // end namespace cryptanalysislib
