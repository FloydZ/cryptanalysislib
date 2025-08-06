#include <iterator>
#include "find.h"


// TODO SIMD optimized version 
// TODO parallel version 

namespace cryptanalysislib {
    template<class ForwardIt, 
             class T = typename std::iterator_traits<ForwardIt>::value_type>
    ForwardIt remove(ForwardIt first,
                     ForwardIt last,
                     const T& value) {
        first = std::find(first, last, value);
        if (first != last)
            for (ForwardIt i = first; ++i != last;)
                if (!(*i == value))
                    *first++ = std::move(*i);
        return first;
    }
    
    template<class ForwardIt,
             class UnaryPred>
    ForwardIt remove_if(ForwardIt first,
                        ForwardIt last,
                        UnaryPred p) {
        first = cryptanalysislib::find_if(first, last, p);
        if (first != last)
            for (ForwardIt i = first; ++i != last;)
                if (!p(*i))
                    *first++ = std::move(*i);
        return first;
    }
    
    template<class InputIt,
             class OutputIt,
             class T = typename std::iterator_traits<InputIt>::value_type>
    constexpr OutputIt remove_copy(InputIt first,
                                   InputIt last,
                                   OutputIt d_first,
                                   const T& value) {
        for (; first != last; ++first)
            if (!(*first == value))
                *d_first++ = *first;
        return d_first;
    }
    
    template<class InputIt,
             class OutputIt,
             class UnaryPred>
    constexpr OutputIt remove_copy_if(InputIt first,
                                      InputIt last,
                                      OutputIt d_first,
                                      UnaryPred p) {
        for (; first != last; ++first) {
            if (!p(*first)) {
                *d_first++ = *first;
            }
        }
        return d_first;
    }
}; // end namespace cryptanalysislib
