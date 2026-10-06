#pragma once

#include <iterator>
#include <algorithm>
#include "thread/thread.h"

/// todo replace ExecPolicy with my parallel policy


namespace cryptanalysislib {
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::forward_iterator<ForwardIt>
#endif
    ForwardIt shift_left(ForwardIt first,
                         ForwardIt last,
                         typename std::iterator_traits<ForwardIt>::difference_type n);

    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<ForwardIt>
#endif
    ForwardIt shift_right(ForwardIt first,
                          ForwardIt last,
                          typename std::iterator_traits<ForwardIt>::difference_type n);

    /// Shifts the elements in the range [first, last) by n positions
    /// Elements are moved forward (towards first) if n is negative,
    /// or backward (towards last) if n is positive
    /// Elements moved before first or beyond last are discarded
    /// 
    /// \tparam ForwardIt Forward iterator type
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift (positive shifts right, negative shifts left)
    /// \return Iterator to the new position of the first element that was not discarded
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<ForwardIt>
#endif
    ForwardIt shift(ForwardIt first,
                    ForwardIt last, 
                    typename std::iterator_traits<ForwardIt>::difference_type n) {
        // positive: towards `last` (= shift_right), negative: towards `first` (= shift_left)
        if (n > 0) {
            return cryptanalysislib::shift_right(first, last, n);
        }

        if (n < 0) {
            cryptanalysislib::shift_left(first, last, -n);
        }

        return first;
    }
    
    /// Shifts elements left by n positions
    /// Elements are moved backward (towards first)
    /// 
    /// \tparam ForwardIt Forward iterator type
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift left
    /// \return Iterator to the new end of the range (last - n)
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::forward_iterator<ForwardIt>
#endif
    ForwardIt shift_left(ForwardIt first,
                         ForwardIt last,
                         typename std::iterator_traits<ForwardIt>::difference_type n) {
        if (n == 0) {
            return last;
        }
        
        if (n >= std::distance(first, last)) {
            // All elements would be shifted out of range
            return first;
        }
        
        // Move elements from [first+n, last) to [first, last-n)
        auto result = std::move(std::next(first, n), last, first);
        return result;
    }
    
    /// Shifts elements right by n positions
    /// Elements are moved forward (towards last)
    /// 
    /// \tparam ForwardIt Forward iterator type
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift right
    /// \return Iterator to the new beginning of the range (first + n)
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<ForwardIt>
#endif
    ForwardIt shift_right(ForwardIt first,
                          ForwardIt last,
                          typename std::iterator_traits<ForwardIt>::difference_type n) {
        using diff_t = typename std::iterator_traits<ForwardIt>::difference_type;
        if (n <= 0) {
            return first;
        }

        diff_t size = 0;
        for (ForwardIt it = first; it != last; ++it) {
            size += 1;
        }

        if (n >= size) {
            // All elements would be shifted out of range
            return last;
        }

        // Move elements from [first, last-n) to [first+n, last), back to front
        ForwardIt src = first;
        for (diff_t i = 0; i < size - n; ++i) {
            ++src;
        }

        ForwardIt dst = last;
        while (src != first) {
            --src;
            --dst;
            *dst = static_cast<typename std::iterator_traits<ForwardIt>::value_type &&>(*src);
        }

        // `dst` is now first + n
        return dst;
    }
    
    /// Shifts elements left by n positions using parallel execution if possible
    /// Elements are moved backward (towards first)
    /// 
    /// \tparam ExecPolicy Execution policy type
    /// \tparam ForwardIt Forward iterator type
    /// \param policy Execution policy
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift left
    /// \return Iterator to the new end of the range (last - n)
    template<class ExecPolicy,
             class ForwardIt>
#if __cplusplus > 201709L
    requires std::random_access_iterator<ForwardIt>
#endif
    ForwardIt shift_left(ExecPolicy&& policy,
                         ForwardIt first,
                         ForwardIt last,
                         typename std::iterator_traits<ForwardIt>::difference_type n) {
        if (n == 0) {
            return last;
        }
        
        if (n >= std::distance(first, last)) {
            // All elements would be shifted out of range
            return first;
        }
        
        if (is_seq<ExecPolicy>(policy)) {
            return shift_left(first, last, n);
        }
        
        // Move elements from [first+n, last) to [first, last-n)
        auto result = std::move(std::next(first, n), last, first);
        return result;
    }
    
    /// Shifts elements right by n positions using parallel execution if possible
    /// Elements are moved forward (towards last)
    /// 
    /// \tparam ExecPolicy Execution policy type
    /// \tparam ForwardIt Forward iterator type
    /// \param policy Execution policy
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift right
    /// \return Iterator to the new beginning of the range (first + n)
    template<class ExecPolicy, class ForwardIt>
#if __cplusplus > 201709L
    requires std::random_access_iterator<ForwardIt>
#endif
    ForwardIt shift_right(ExecPolicy&& policy,
                          ForwardIt first, 
                          ForwardIt last,
                          typename std::iterator_traits<ForwardIt>::difference_type n) {
        if (n == 0) {
            return first;
        }
        
        if (n >= std::distance(first, last)) {
            // All elements would be shifted out of range
            return last;
        }
        
        // NOTE: not parallelized yet
        (void)policy;
        return cryptanalysislib::shift_right(first, last, n);
    }
}; // end namespace cryptanalysislib
