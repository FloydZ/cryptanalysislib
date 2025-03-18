#pragma once

#include <iterator>
#include <cstdint>

namespace cryptanalysislib {
    struct AlgorithmSearchConfig {
    public:
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmSearchConfig algorithmSearchConfig{};

    /// \tparam
    template<class ForwardIt1,
             class ForwardIt2>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt1> &&
                 std::forward_iterator<ForwardIt2>
#endif
    constexpr
    ForwardIt1 search(ForwardIt1 first,
                      ForwardIt1 last,
                      ForwardIt2 s_first, 
                      ForwardIt2 s_last) noexcept {
    	while (true) {
    		ForwardIt1 it = first;
    		for (ForwardIt2 s_it = s_first;; ++it, ++s_it) {
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
    
    template<class ForwardIt1, 
             class ForwardIt2, 
             class BinaryPred>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt1> &&
                 std::forward_iterator<ForwardIt2> && 
    		     std::regular_invocable<BinaryPred, bool>
#endif
    constexpr
    ForwardIt1 search(ForwardIt1 first, 
                      ForwardIt1 last,
                      ForwardIt2 s_first, 
                      ForwardIt2 s_last, 
                      BinaryPred p) noexcept {
    	while (true) {
    		ForwardIt1 it = first;
    		for (ForwardIt2 s_it = s_first;; ++it, ++s_it) {
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
    
    template<class ForwardIt,
             class Size,
             class T = typename std::iterator_traits<ForwardIt>::value_type>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
    constexpr
    ForwardIt search_n(ForwardIt first, 
                       ForwardIt last,
                       Size count, 
                       const T &value) noexcept {
    	if (count <= 0) {
    		return first;
        }
    
    	for (; first != last; ++first) {
    		if (!(*first == value))
    			continue;
    
    		ForwardIt candidate = first;
    
    		for (Size cur_count = 1; true; ++cur_count) {
    			if (cur_count >= count)
    				return candidate;// success
    
    			++first;
    			if (first == last)
    				return last;// exhausted the list
    
    			if (!(*first == value))
    				break;// too few in a row
    		}
    	}
    	return last;
    }
    
    template<class ForwardIt, 
             class Size,
             class T = typename std::iterator_traits<ForwardIt>::value_type,
             class BinaryPred>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt> &&
    		     std::regular_invocable<BinaryPred, bool>
#endif
    constexpr
    ForwardIt search_n(ForwardIt first, 
                       ForwardIt last, 
                       Size count, 
                       const T &value,
                       BinaryPred p) noexcept {
    	if (count <= 0) {
    		return first;
        }
    
    	for (; first != last; ++first) {
    		if (!p(*first, value))
    			continue;
    
    		ForwardIt candidate = first;
    
    		for (Size cur_count = 1; true; ++cur_count) {
    			if (cur_count >= count)
    				return candidate;// success
    
    			++first;
    			if (first == last)
    				return last;// exhausted the list
    
    			if (!p(*first, value))
    				break;// too few in a row
    		}
    	}
    	return last;
    }
}; // end namespace
