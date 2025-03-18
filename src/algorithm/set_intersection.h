#pragma once 

#include <iterator>
#include <cstdint>

// TODO simd version which also supports 

namespace cryptanalysislib {
    struct AlgorithmSetIntersectionConfig {
    public:
        const size_t aligned_instructions = false;
    	const uint32_t min_size_simd = 32;
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmSetIntersectionConfig algorithmSetIntersectionConfig{};


    /// NOTE: must be sorted
    template<class InputIt1, 
             class InputIt2, 
             class OutputIt, 
             class Compare>
#if __cplusplus > 201709L
    	requires std::forward_iterator<InputIt1> && 
                 std::forward_iterator<InputIt2> && 
                 std::forward_iterator<OutputIt>
#endif
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
    
    /// NOTE: must be sorted
    template<class InputIt1, 
             class InputIt2, 
             class OutputIt>
#if __cplusplus > 201709L
    	requires std::forward_iterator<InputIt1> && 
                 std::forward_iterator<InputIt2> && 
                 std::forward_iterator<OutputIt>
#endif
    constexpr
    OutputIt set_intersection(InputIt1 first1,
                              InputIt1 last1,
                              InputIt2 first2,
                              InputIt2 last2, 
                              OutputIt d_first) {
        return set_intersection(first1, last1, first2, last2, d_first, std::less{});
    }

}; // end namespace
