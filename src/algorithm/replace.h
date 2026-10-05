#pragma once
#include <iterator>

namespace cryptanalysislib {
    namespace internal {
        // TODO implement SIMD optimized version
    };

    /// Replaces all elements equal to old_value with new_value in a range
    ///
    /// \tparam ForwardIt type of the forward iterator
    /// \tparam T type of the values being replaced, deduced from iterator
    /// \param first[in]: iterator to the first element in the range
    /// \param last[in]: iterator to one past the last element in the range
    /// \param old_value[in]: value to be replaced
    /// \param new_value[in]: value to replace with
    template<class ForwardIt,
             class T = typename std::iterator_traits<ForwardIt>::value_type>
    void replace(ForwardIt first,
                 ForwardIt last,
                 const T& old_value,
                 const T& new_value) {
        for (; first != last; ++first)
            if (*first == old_value)
                *first = new_value;
    }
    
    /// Replaces all elements satisfying a predicate with new_value in a range
    ///
    /// \tparam ForwardIt type of the forward iterator
    /// \tparam UnaryPred type of the unary predicate function
    /// \tparam T type of the values being replaced, deduced from iterator
    /// \param first[in]: iterator to the first element in the range
    /// \param last[in]: iterator to one past the last element in the range
    /// \param p[in]: unary predicate which returns true for elements to be replaced
    /// \param new_value[in]: value to replace with
    template<class ForwardIt, class UnaryPred,
             class T = typename std::iterator_traits<ForwardIt>::value_type>
    void replace_if(ForwardIt first, ForwardIt last,
                    UnaryPred p, const T& new_value)
    {
        for (; first != last; ++first)
            if (p(*first))
                *first = new_value;
    }

    template<class InputIt, class OutputIt, class T>
    OutputIt replace_copy(InputIt first,
                          InputIt last,
                          OutputIt d_first,
                          const T& old_value,
                          const T& new_value) {
        for (; first != last; ++first)
            *d_first++ = (*first == old_value) ? new_value : *first;
        return d_first;
    }

    template<class InputIt,
             class OutputIt,
             class UnaryPred,
             class T = typename std::iterator_traits<InputIt>::value_type>
    OutputIt replace_copy_if(InputIt first, 
                             InputIt last,
                             OutputIt d_first,
                             UnaryPred p,
                             const T& new_value) {
        for (; first != last; ++first)
            *d_first++ = p(*first) ? new_value : *first;
        return d_first;
    }
}; // end namespace cryptanalysislib
