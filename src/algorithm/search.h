#pragma once 
template<class ForwardIt1, class ForwardIt2>
constexpr //< since C++20
ForwardIt1 search(ForwardIt1 first, ForwardIt1 last,
                  ForwardIt2 s_first, ForwardIt2 s_last)
{
    while (true)
    {
        ForwardIt1 it = first;
        for (ForwardIt2 s_it = s_first; ; ++it, ++s_it)
        {
            if (s_it == s_last)
                return first;
            if (it == last)
                return last;
            if (!(*it == *s_it))
                break;
        }
        ++first;
    }
}

template<class ForwardIt1, class ForwardIt2, class BinaryPred>
constexpr //< since C++20
ForwardIt1 search(ForwardIt1 first, ForwardIt1 last,
                  ForwardIt2 s_first, ForwardIt2 s_last, BinaryPred p)
{
    while (true)
    {
        ForwardIt1 it = first;
        for (ForwardIt2 s_it = s_first; ; ++it, ++s_it)
        {
            if (s_it == s_last)
                return first;
            if (it == last)
                return last;
            if (!p(*it, *s_it))
                break;
        }
        ++first;
    }
}

template<class ForwardIt, class Size,
         class T = typename std::iterator_traits<ForwardIt>::value_type>
ForwardIt search_n(ForwardIt first, ForwardIt last, Size count, const T& value)
{
    if (count <= 0)
        return first;
 
    for (; first != last; ++first)
    {
        if (!(*first == value))
            continue;
 
        ForwardIt candidate = first;
 
        for (Size cur_count = 1; true; ++cur_count)
        {
            if (cur_count >= count)
                return candidate; // success
 
            ++first;
            if (first == last)
                return last; // exhausted the list
 
            if (!(*first == value))
                break; // too few in a row
        }
    }
    return last;
}

template<class ForwardIt, class Size,
         class T = typename std::iterator_traits<ForwardIt>::value_type,
         class BinaryPred>
ForwardIt search_n(ForwardIt first, ForwardIt last, Size count, const T& value,
                   BinaryPred p)
{
    if (count <= 0)
        return first;
 
    for (; first != last; ++first)
    {
        if (!p(*first, value))
            continue;
 
        ForwardIt candidate = first;
 
        for (Size cur_count = 1; true; ++cur_count)
        {
            if (cur_count >= count)
                return candidate; // success
 
            ++first;
            if (first == last)
                return last; // exhausted the list
 
            if (!p(*first, value))
                break; // too few in a row
        }
    }
    return last;
}
