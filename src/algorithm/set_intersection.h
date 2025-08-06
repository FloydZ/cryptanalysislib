#pragma once 

#include <iterator>
#include <cstdint>

// TODO simd version 
// TODO parrallel version

namespace cryptanalysislib {
    struct AlgorithmSetIntersectionConfig {
    public:
        const size_t aligned_instructions = false;
    	const uint32_t min_size_simd = 32;
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmSetIntersectionConfig algorithmSetIntersectionConfig{};


    /// Computes the intersection of two sorted ranges
    ///
    /// \tparam InputIt1 type of the first input iterator
    /// \tparam InputIt2 type of the second input iterator
    /// \tparam OutputIt type of the output iterator
    /// \tparam Compare type of the comparison function
    /// 
    /// NOTE: Input ranges must be sorted according to the same ordering criterion
    template<class InputIt1, 
             class InputIt2, 
             class OutputIt, 
             class Compare>
#if __cplusplus > 201709L
    	requires std::forward_iterator<InputIt1> && 
                 std::forward_iterator<InputIt2> && 
                 std::forward_iterator<OutputIt>
#endif
    /// Constructs a sorted range consisting of elements that are found in both sorted input ranges
    ///
    /// \param first1[in]: iterator to the beginning of the first range
    /// \param last1[in]: iterator to the end of the first range
    /// \param first2[in]: iterator to the beginning of the second range
    /// \param last2[in]: iterator to the end of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: comparison function object
    /// \return iterator to the end of the constructed range
    constexpr
    OutputIt set_intersection(InputIt1 first1, 
                              InputIt1 last1,
                              InputIt2 first2,
                              InputIt2 last2, 
                              OutputIt d_first, 
                              Compare comp) noexcept {
        while (first1 != last1 && first2 != last2) {
            if (comp(*first1, *first2)) {
                ++first1;
            } else {
                if (!comp(*first2, *first1)) {
                    // *first1 and *first2 are equivalent.
                    *d_first++ = *first1++; 
                }
                ++first2;
            }
        }
        return d_first;
    }
    
    /// Computes the intersection of two sorted ranges using the less operator
    ///
    /// \tparam InputIt1 type of the first input iterator
    /// \tparam InputIt2 type of the second input iterator
    /// \tparam OutputIt type of the output iterator
    /// 
    /// NOTE: Input ranges must be sorted in ascending order
    template<class InputIt1, 
             class InputIt2, 
             class OutputIt>
#if __cplusplus > 201709L
    	requires std::forward_iterator<InputIt1> && 
                 std::forward_iterator<InputIt2> && 
                 std::forward_iterator<OutputIt>
#endif
    /// Constructs a sorted range consisting of elements that are found in both sorted input ranges
    ///
    /// \param first1[in]: iterator to the beginning of the first range
    /// \param last1[in]: iterator to the end of the first range
    /// \param first2[in]: iterator to the beginning of the second range
    /// \param last2[in]: iterator to the end of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \return iterator to the end of the constructed range
    constexpr
    OutputIt set_intersection(InputIt1 first1,
                              InputIt1 last1,
                              InputIt2 first2,
                              InputIt2 last2, 
                              OutputIt d_first) {
        return set_intersection(first1, last1, first2, last2, d_first, std::less{});
    }

}; // end namespace
