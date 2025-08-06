#pragma once

#include <utility>

/// TODO simd version implementation
/// TODO parallel version implementation

namespace cryptanalysislib {
    template<class T>
    constexpr 
    std::pair<const T&, const T&> minmax(const T& a,
                                         const T& b) noexcept {
        return (b < a) ? std::pair<const T&, const T&>(b, a)
                       : std::pair<const T&, const T&>(a, b);
    }

    template<class T, class Compare>
    constexpr 
    std::pair<const T&, const T&> minmax(const T& a,
                                         const T& b,
                                         Compare comp) noexcept {
        return comp(b, a) ? std::pair<const T&, const T&>(b, a)
                          : std::pair<const T&, const T&>(a, b);
    }
    
    template<class T>
    constexpr 
    std::pair<T, T> minmax(std::initializer_list<T> ilist) noexcept {
        auto p = minmax_element(ilist.begin(), ilist.end());
        return std::pair(*p.first, *p.second);
    }

    template<class ForwardIt>
    constexpr 
    std::pair<ForwardIt, ForwardIt>
        minmax_element(ForwardIt first, ForwardIt last) {
        using value_type = typename std::iterator_traits<ForwardIt>::value_type;
        return minmax_element(first, last, std::less<value_type>());
    }
    
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
