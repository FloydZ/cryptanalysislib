#pragma once

#include <iterator>
#include <algorithm>
#include "thread/thread.h"

/// todo replace ExecPolicy with my parallel policy


namespace cryptanalysislib {
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
    requires std::forward_iterator<ForwardIt>
#endif
    ForwardIt shift(ForwardIt first,
                    ForwardIt last, 
                    typename std::iterator_traits<ForwardIt>::difference_type n) {
        if (n == 0) {
            return first;
        }
        
        if (n > 0) {
            // Right shift (towards the end)
            const auto mid = first;
            std::advance(first, std::min(n, std::distance(first, last)));
            // Move elements from [first+n, last) to [first, last-n)
            return std::move(first, last, mid);
        } else {
            // Left shift (towards the beginning)
            // Convert negative n to positive for easier handling
            const auto abs_n = -n;
            const auto d = std::distance(first, last);
            
            if (abs_n >= d) {
                // All elements would be shifted out of range
                return last;
            }
            
            // Move elements from [first, last-abs_n) to [first+abs_n, last)
            auto mid = last;
            std::advance(mid, -abs_n);
            std::move(first, mid, std::next(first, abs_n));
            return first;
        }
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
    requires std::forward_iterator<ForwardIt>
#endif
    ForwardIt shift_right(ForwardIt first,
                          ForwardIt last,
                          typename std::iterator_traits<ForwardIt>::difference_type n) {
        if (n == 0) {
            return first;
        }
        
        if (n >= std::distance(first, last)) {
            // All elements would be shifted out of range
            return last;
        }
        
        // Move elements from [first, last-n) to [first+n, last)
        auto it = last;
        while (n > 0) {
            --it;
            --n;
        }
        
        std::move_backward(first, it, last);
        return std::next(first, n);
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
        
        if (is_seq<ExecPolicy>(policy)) {
            return shift_right(first, last, n);
        }

        // LOL todo not correct
        
        // Move elements from [first, last-n) to [first+n, last)
        auto it = last;
        while (n > 0) {
            --it;
            --n;
        }
        
        std::move_backward(first, it, last);
        return std::next(first, n);
    }
}; // end namespace cryptanalysislib
