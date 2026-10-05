#pragma once 

namespace cryptanalysislib {

    /// TODO doc
    template<class ForwardIt>
    constexpr //< since C++20
    ForwardIt is_sorted_until(ForwardIt first,
                              ForwardIt last) {
        return is_sorted_until(first, last, std::less<>());
    }

    /// TODO doc
    template<class ForwardIt,
             class Compare>
    constexpr
    ForwardIt is_sorted_until(ForwardIt first,
                              ForwardIt last,
                              Compare comp) {
        if (first != last) {
            ForwardIt next = first;
            while (++next != last) {
                if (comp(*next, *first)) {
                    return next;
                }
                first = next;
            }
        }

        return last;
    }
    
    /// TODO doc
    template<class ForwardIt>
    bool is_sorted(ForwardIt first,
                   ForwardIt last) {
        return is_sorted_until(first, last) == last;
    }
   
    /// TODO doc
    template<class ForwardIt, class Compare>
    bool is_sorted(ForwardIt first,
                   ForwardIt last,
                   Compare comp) {
        return is_sorted_until(first, last, comp) == last;
    }




template<typename RandomIt>
constexpr //< since C++20
void partial_sort(RandomIt first, RandomIt middle, RandomIt last)
{
    typedef typename std::iterator_traits<RandomIt>::value_type VT;
    std::partial_sort(first, middle, last, std::less<VT>());
}
namespace impl
{
    template<typename RandomIt, typename Compare>
    constexpr //< since C++20
    void sift_down(RandomIt first, RandomIt last, const Compare& comp)
    {
        // sift down element at “first”
        const auto length = static_cast<std::size_t>(last - first);
        std::size_t current = 0;
        std::size_t next = 2;
        while (next < length)
        {
            if (comp(*(first + next), *(first + (next - 1))))
                --next;
            if (!comp(*(first + current), *(first + next)))
                return;
            std::iter_swap(first + current, first + next);
            current = next;
            next = 2 * current + 2;
        }
        --next;
        if (next < length && comp(*(first + current), *(first + next)))
            std::iter_swap(first + current, first + next);
    }
 
    template<typename RandomIt, typename Compare>
    constexpr //< since C++20
    void heap_select(RandomIt first, RandomIt middle, RandomIt last, const Compare& comp)
    {
        std::make_heap(first, middle, comp);
        for (auto i = middle; i != last; ++i)
        {
            if (comp(*i, *first))
            {
                std::iter_swap(first, i);
                sift_down(first, middle, comp);
            }
        }
    }
} // namespace impl
 
template<typename RandomIt, typename Compare>
constexpr //< since C++20
void partial_sort(RandomIt first, RandomIt middle, RandomIt last, Compare comp)
{
    impl::heap_select(first, middle, last, comp);
    std::sort_heap(first, middle, comp);
}


};
