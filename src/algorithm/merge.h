#pragma once 

/// TODO simd version 
/// TODO parallel version 

namespace cryptanalysislib {
    
    /// TODO doc
    template<class InputIt1,
             class InputIt2,
             class OutputIt>
    OutputIt merge(InputIt1 first1,
                   InputIt1 last1,
                   InputIt2 first2,
                   InputIt2 last2,
                   OutputIt d_first) {
        for (; first1 != last1; ++d_first) {
            if (first2 == last2)
                return std::copy(first1, last1, d_first);
     
            if (*first2 < *first1) {
                *d_first = *first2;
                ++first2;
            } else {
                *d_first = *first1;
                ++first1;
            }
        }

        return std::copy(first2, last2, d_first);
    }
   
    /// TODO doc
    template<class InputIt1,
             class InputIt2,
             class OutputIt,
             class Compare>
    OutputIt merge(InputIt1 first1,
                   InputIt1 last1,
                   InputIt2 first2,
                   InputIt2 last2,
                   OutputIt d_first,
                   Compare comp) {
        for (; first1 != last1; ++d_first) {
            if (first2 == last2)
                return std::copy(first1, last1, d_first);
     
            if (comp(*first2, *first1)) {
                *d_first = *first2;
                ++first2;
            } else {
                *d_first = *first1;
                ++first1;
            }
        }

        return std::copy(first2, last2, d_first);
    }
}; // end namespace cryptanalysislib
