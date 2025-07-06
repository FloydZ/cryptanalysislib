#pragma once

#include <iterator>
#include <cstdint>

#include "simd/simd.h"
#include "algorithm/bits/popcount.h"
#include "algorithm/algorithm.h"

namespace cryptanalysislib {
    struct AlgorithmSearchConfig {
    public:
    	const uint32_t min_size_per_thread = 16384;

        const bool aligned_instructions = false;
    };
    constexpr static AlgorithmSearchConfig algorithmSearchConfig{};

	namespace internal {

		/// \tparam T
		/// \param data
		/// \param n
		/// \param val
		/// \return the position of the first element == val for n  elements
		template<typename T,
				 const AlgorithmSearchConfig &config=algorithmSearchConfig>
#if __cplusplus > 201709L
			requires std::unsigned_integral<T>
#endif
		constexpr size_t search_n_uXX_simd(const T *data,
									   const size_t n,
									   const T val) noexcept {
			using S = SIMDSelector<T>;
			using U = S::limb_type;

			const auto t = S::set1(val);
            
			size_t i = 0;
            // TODO optimize the code for the case n < 4, 8, 16
			for (; (i+S::LIMBS) <= n; i+=n) {
				const auto d = S::template load<config.aligned_instructions>(data + i);
				const U s = d == t;
				if (popcount::popcount(s) == n) [[unlikely]] {
					return i + ffs<T>(s) - 1u;
				}
			}

            // tailmanagment
			for (; i+n <= n; i++) {
				if (data[i] != val) {
                    continue;
				}

    		    for (size_t cur_count = 1; true; ++cur_count) {
    		    	if (cur_count >= n)
    		    		return i;
                    
                    // exhausted the list
    		    	if (i + cur_count == n) {
    		    		return n;
                    }

                    // too few in a row
    		    	if (!(data[i + cur_count] == val)) {
    		    		break;
                    }
    		    }
			}

			return i;
		}
	}// end namespace internal

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
             class Size>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
    constexpr
    ForwardIt search_n(ForwardIt first, 
                       ForwardIt last,
                       Size count, 
                       const typename std::iterator_traits<ForwardIt>::value_type &value) noexcept {
        using T = typename std::iterator_traits<ForwardIt>::value_type;
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
             class BinaryPred>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt> &&
    		     std::regular_invocable<BinaryPred, bool>
#endif
    constexpr
    ForwardIt search_n(ForwardIt first, 
                       ForwardIt last, 
                       Size count, 
                       const typename std::iterator_traits<ForwardIt>::value_type &value,
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


	template<class ExecPolicy,
			 class RandIt,
             class Size,
             class BinaryPred,
			 const AlgorithmSearchConfig &config = algorithmSearchConfig>
#if __cplusplus > 201709L
	requires std::random_access_iterator<RandIt>
#endif
	RandIt search_n(ExecPolicy &&policy,
                    RandIt first, 
                    RandIt last, 
                    Size count, 
                    const typename std::iterator_traits<RandIt>::value_type &value,
                    BinaryPred &&p) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;
		using diff_t = typename std::iterator_traits<RandIt>::difference_type;
		const diff_t size = std::distance(first, last);
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return search_n<RandIt, Size, BinaryPred>(first, last, count, value, p);
		}

		std::atomic<diff_t> found(size);

		internal::parallel_chunk_for_1_wait(std::forward<ExecPolicy>(policy), first, last,
			[&first, &found, &count, &value, p](RandIt chunk_first,
									  RandIt chunk_last)
									  __attribute__((always_inline)) {
				if (std::distance(first, chunk_first) > found) {
					// already found by another task
					return;
				}

				RandIt chunk_res = search_n<RandIt, Size, BinaryPred>
                    (chunk_first, chunk_last, count, value, p);

				if (chunk_res != chunk_last) {
					const diff_t k = std::distance(first, chunk_res);
					for (diff_t old = found; k < old; old = found) {
						found.compare_exchange_weak(old, k);
					}
				}
			}, (void*)nullptr,
			8,
			nthreads);

		// use small tasks so later ones may exit early if item is already found
		return found == size ? last : first + found;
	}
}; // end namespace
