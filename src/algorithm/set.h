#pragma once 

#include <atomic>
#include <future>
#include <vector>

#include "copy.h"
#include "simd/simd.h"
#include "algorithm/bits/ffs.h"
#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    namespace internal {
        /// NOTE: tests in `tests/algorithm/set.cpp`
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
                cryptanalysislib::copy(data1, data1 + n1, dest);
                return n1;
            }
            
            using S = SIMDSelector<T>;
            // lane mask with one bit set per SIMD lane
            // NOTE: 64 bit: with AVX-512, uint8_t has 64 lanes; the former
            //     32 bit mask dropped the upper half (wrong results).
            static_assert(S::LIMBS <= 64);
            constexpr uint64_t full_mask = S::LIMBS >= 64 ? uint64_t(-1ull) : ((1ull << S::LIMBS) - 1ull);
            
            size_t i1 = 0, i2 = 0, dest_idx = 0;
            
            while (i1 < n1) {
                // If we reached the end of data2, copy remaining elements from data1
                if (i2 == n2) {
                    cryptanalysislib::copy(data1 + i1, data1 + n1, dest + dest_idx);
                    dest_idx += (n1 - i1);
                    break;
                }
                
                // If we have enough elements left in data1 for a SIMD comparison
                if (i1 + S::LIMBS <= n1) {
                    // Load a SIMD-width chunk of data1
                    const auto chunk1 = S::template load<false>(data1 + i1);
                    // Create a SIMD vector with the current data2 element replicated
                    const auto val2 = S::set1(data2[i2]);
                    
                    // Compare if all elements in chunk1 are less than data2[i2]
                    const uint64_t less_mask = uint64_t(S::lt(chunk1, val2)) & full_mask;
                    
                    if (less_mask == full_mask) {
                        // All elements in chunk1 are less than data2[i2], 
                        // so they all belong in the difference
                        S::template store<false>(dest + dest_idx, chunk1);
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

        /// NOTE: tests in `tests/algorithm/set.cpp`
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
            // lane mask with one bit set per SIMD lane
            // NOTE: 64 bit: with AVX-512, uint8_t has 64 lanes; the former
            //     32 bit masks dropped the upper half (wrong results).
            static_assert(S::LIMBS <= 64);
            constexpr uint64_t full_mask = S::LIMBS >= 64 ? uint64_t(-1ull) : ((1ull << S::LIMBS) - 1ull);
            
            size_t i1 = 0, i2 = 0;
            
            // Process elements with SIMD when possible
            while (i1 < n1 && i2 < n2) {
                // If we have enough elements left in data1 for a SIMD comparison
                if (i1 + S::LIMBS <= n1) {
                    // Load a SIMD-width chunk of data1
                    const auto chunk1 = S::template load<false>(data1 + i1);
                    // Create a SIMD vector with the current data2 element replicated
                    const auto val2 = S::set1(data2[i2]);
                    
                    // Compare if any element in chunk1 equals val2
                    const uint64_t mask = uint64_t(S::eq(chunk1, val2)) & full_mask;
                    
                    if (mask) {
                        // Found a match, advance to next data2 element
                        ++i2;
                        // Advance i1 to first element after the match
                        const size_t match_pos = ffs<uint64_t>(mask) - 1u;
                        i1 += match_pos + 1;
                    } else {
                        // No match in this chunk
                        // Check if all elements in chunk1 are less than data2[i2]
                        const uint64_t less_mask = uint64_t(S::lt(chunk1, val2)) & full_mask;
                        
                        if (less_mask == full_mask) {
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
            // NOTE: was marked "definitely not correct": the 32 bit masks were
            //     wrong for 64 lanes (uint8_t with AVX-512); the algorithm
            //     itself agrees with std::includes (tests/algorithm/set.cpp).
            // If we've processed all elements in data2, it's a subset
            return i2 == n2;
        }
    }

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
                      std::contiguous_iterator<InputIt1> &&
                      std::contiguous_iterator<InputIt2>) {
            if (first2 == last2) {
                return true;
            }
            if (first1 == last1) {
                return false;
            }
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
                      std::contiguous_iterator<InputIt1> &&
                      std::contiguous_iterator<InputIt2>) {
            const size_t n1 = static_cast<size_t>(std::distance(first1, last1));
            const size_t n2 = static_cast<size_t>(std::distance(first2, last2));
            
            // Create temporary buffer for results
            std::vector<T> temp_buffer(n1);
            
            // Call SIMD implementation
            const size_t result_size = internal::set_difference_uXX_simd<T>(
                n1 ? &(*first1) : nullptr, n1,
                n2 ? &(*first2) : nullptr, n2,
                temp_buffer.data()
            );
            
            // Copy result to output iterator
            return cryptanalysislib::copy(temp_buffer.data(), temp_buffer.data() + result_size, d_first);
        }
        
        // Standard implementation for non-SIMD compatible types
        while (first1 != last1) {
            if (first2 == last2) {
                return cryptanalysislib::copy(first1, last1, d_first);
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
                return cryptanalysislib::copy(first1, last1, d_first);
     
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
                return cryptanalysislib::copy(first1, last1, d_first);
     
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
        return cryptanalysislib::copy(first2, last2, d_first);
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
                return cryptanalysislib::copy(first1, last1, d_first);
     
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
        return cryptanalysislib::copy(first2, last2, d_first);
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
                return cryptanalysislib::copy(first1, last1, d_first);
     
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
        return cryptanalysislib::copy(first2, last2, d_first);
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
                return cryptanalysislib::copy(first1, last1, d_first);
     
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
        return cryptanalysislib::copy(first2, last2, d_first);
    }

    /// Configuration for the parallel set algorithms
    struct AlgorithmSetConfig : public AlgorithmConfig {
        constexpr static size_t min_size_per_thread = 1u << 15u;
    };
    constexpr static AlgorithmSetConfig algorithmSetConfig;

    namespace internal {
        /// \return first position `p` in [lo, hi) with !(a[p] < v), hi if none
        template<class RandIt, class T>
        constexpr size_t set_lower_pos(RandIt a, size_t lo, size_t hi, const T &v) noexcept {
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (a[mid] < v) { lo = mid + 1; } else { hi = mid; }
            }
            return lo;
        }

        /// \return first position `p` in [lo, hi) with v < a[p], hi if none
        template<class RandIt, class T>
        constexpr size_t set_upper_pos(RandIt a, size_t lo, size_t hi, const T &v) noexcept {
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (v < a[mid]) { hi = mid; } else { lo = mid + 1; }
            }
            return lo;
        }

        /// splits two sorted ranges into `nthreads` pairs of subranges with
        /// disjoint, increasing value ranges: range 1 is cut into equal
        /// chunks, every cut is moved behind the run of equal elements it
        /// falls into (so equal elements are never split: multiset semantics
        /// are kept), and range 2 is cut at the lower bound of the same value.
        /// \return cut positions in both ranges, nthreads + 1 each
        template<class RandIt1, class RandIt2>
        static std::pair<std::vector<size_t>, std::vector<size_t>>
        set_parallel_cuts(RandIt1 first1, const size_t n1,
                          RandIt2 first2, const size_t n2,
                          const uint32_t nthreads) noexcept {
            std::vector<size_t> c1(nthreads + 1), c2(nthreads + 1);
            c1[0] = c2[0] = 0;
            c1[nthreads] = n1; c2[nthreads] = n2;
            for (uint32_t k = 1; k < nthreads; ++k) {
                size_t c = (n1 * k) / nthreads;
                if (c < c1[k - 1]) { c = c1[k - 1]; }
                if ((c > 0) && (c < n1)) { c = set_upper_pos(first1, c, n1, first1[c - 1]); }
                c1[k] = c;
                size_t d = (c < n1) ? set_lower_pos(first2, 0, n2, first1[c]) : n2;
                if (d < c2[k - 1]) { d = c2[k - 1]; }
                c2[k] = d;
            }
            return {c1, c2};
        }

        /// runs the sequential set operation `op` on every pair of subranges
        /// (see `set_parallel_cuts`) in parallel, each into a local buffer,
        /// which are concatenated into `d_first`.
        template<class ExecPolicy, class RandIt1, class RandIt2, class OutIt, class Op>
        static OutIt set_op_parallel(ExecPolicy &&policy,
                                     RandIt1 first1, const size_t n1,
                                     RandIt2 first2, const size_t n2,
                                     OutIt d_first, Op op,
                                     const uint32_t nthreads) noexcept {
            using T = typename std::iterator_traits<RandIt1>::value_type;
            // NOTE: no structured binding, clang < 19 cannot capture them with -fopenmp
            const auto cuts = set_parallel_cuts(first1, n1, first2, n2, nthreads);
            const std::vector<size_t> &c1 = cuts.first, &c2 = cuts.second;
            auto &task_pool = *policy.pool();

            std::vector<std::vector<T>> parts(nthreads);
            {
                std::vector<std::future<void>> futures;
                for (uint32_t k = 0; k < nthreads; ++k) {
                    futures.emplace_back(task_pool.submit([&, k]() {
                        std::vector<T> &out = parts[k];
                        out.resize((c1[k + 1] - c1[k]) + (c2[k + 1] - c2[k]));
                        auto e = op(first1 + c1[k], first1 + c1[k + 1],
                                    first2 + c2[k], first2 + c2[k + 1], out.begin());
                        out.resize(static_cast<size_t>(e - out.begin()));
                    }));
                }
                for (auto &f : futures) { f.wait(); }
            }

            std::vector<size_t> off(nthreads + 1, 0);
            for (uint32_t k = 0; k < nthreads; ++k) { off[k + 1] = off[k] + parts[k].size(); }

            std::vector<std::future<void>> futures;
            for (uint32_t k = 0; k < nthreads; ++k) {
                futures.emplace_back(task_pool.submit([&, k]() {
                    cryptanalysislib::copy(parts[k].begin(), parts[k].end(), d_first + off[k]);
                }));
            }
            for (auto &f : futures) { f.wait(); }
            return d_first + off[nthreads];
        }
    } // end namespace internal

#define CRYPTANALYSISLIB_SET_PARALLEL(NAME)                                              \
    /** parallel version of `NAME`, see `internal::set_op_parallel` */                   \
    template<class ExecPolicy, class RandIt1, class RandIt2, class RandIt3,               \
             const AlgorithmSetConfig &config = algorithmSetConfig>                        \
        requires std::random_access_iterator<RandIt1> &&                                 \
                 std::random_access_iterator<RandIt2> &&                                 \
                 std::random_access_iterator<RandIt3>                                    \
    RandIt3 NAME(ExecPolicy &&policy,                                                     \
                 RandIt1 first1, RandIt1 last1,                                           \
                 RandIt2 first2, RandIt2 last2,                                           \
                 RandIt3 d_first) noexcept {                                              \
        const size_t n1 = static_cast<size_t>(std::distance(first1, last1));              \
        const size_t n2 = static_cast<size_t>(std::distance(first2, last2));              \
        const uint32_t nthreads = should_par(policy, config, n1 + n2);                    \
        if (is_seq<ExecPolicy>(policy) || nthreads < 2 || n1 < nthreads) {                \
            return cryptanalysislib::NAME(first1, last1, first2, last2, d_first);         \
        }                                                                                 \
        return internal::set_op_parallel(policy, first1, n1, first2, n2, d_first,         \
            [](auto a1, auto a2, auto b1, auto b2, auto out) {                            \
                return cryptanalysislib::NAME(a1, a2, b1, b2, out);                       \
            }, nthreads);                                                                 \
    }

    CRYPTANALYSISLIB_SET_PARALLEL(set_difference)
    CRYPTANALYSISLIB_SET_PARALLEL(set_intersection)
    CRYPTANALYSISLIB_SET_PARALLEL(set_union)
    CRYPTANALYSISLIB_SET_PARALLEL(set_symmetric_difference)
#undef CRYPTANALYSISLIB_SET_PARALLEL

    /// parallel `includes`: every pair of subranges (see
    /// `internal::set_parallel_cuts`) is checked in parallel
    template<class ExecPolicy, class RandIt1, class RandIt2,
             const AlgorithmSetConfig &config = algorithmSetConfig>
        requires std::random_access_iterator<RandIt1> &&
                 std::random_access_iterator<RandIt2>
    bool includes(ExecPolicy &&policy,
                  RandIt1 first1, RandIt1 last1,
                  RandIt2 first2, RandIt2 last2) noexcept {
        const size_t n1 = static_cast<size_t>(std::distance(first1, last1));
        const size_t n2 = static_cast<size_t>(std::distance(first2, last2));
        const uint32_t nthreads = should_par(policy, config, n1 + n2);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2 || n1 < nthreads) {
            return cryptanalysislib::includes(first1, last1, first2, last2);
        }

        // NOTE: no structured binding, clang < 19 cannot capture them with -fopenmp
        const auto cuts = internal::set_parallel_cuts(first1, n1, first2, n2, nthreads);
        const std::vector<size_t> &c1 = cuts.first, &c2 = cuts.second;
        auto &task_pool = *policy.pool();
        std::atomic<bool> ok{true};
        std::vector<std::future<void>> futures;
        for (uint32_t k = 0; k < nthreads; ++k) {
            futures.emplace_back(task_pool.submit([&, k]() {
                if (!cryptanalysislib::includes(first1 + c1[k], first1 + c1[k + 1],
                                                first2 + c2[k], first2 + c2[k + 1])) {
                    ok.store(false, std::memory_order_relaxed);
                }
            }));
        }
        for (auto &f : futures) { f.wait(); }
        return ok.load(std::memory_order_relaxed);
    }
}; // end namespace cryptanalysislib
