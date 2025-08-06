#ifndef CRYPTANALYSISLIB_ALGORITHM_FIND_H
#define CRYPTANALYSISLIB_ALGORITHM_FIND_H

#include <iterator>

#include "thread/thread.h"
#include "algorithm/algorithm.h"
#include "algorithm/bits/ffs.h"
#include "simd/simd.h"
#include "search/search.h"

// TODO docs for find_first_of and adjacent_find 
// TODO parallel versions of `find_first_of` and `adjacent_find`

namespace cryptanalysislib {
    /// Configuration for find algorithms 
    // TODO somehow this yields an anonymous unitiialized field element
	struct AlgorithmFindConfig /* : public AlgorithmConfig */ {
		// NOTE multithreaded find is extremly slow
		const size_t min_size_per_thread = 1048576u;

		const bool aligned_instructions = false;

		// NOTE: not really implementable, as `find_if`
		// is based on a predicate
		const bool assume_sorted = false;
		const bool use_interpolation_search = false;
	};

	constexpr static AlgorithmFindConfig algorithmFindConfig;

	namespace internal {

		/// SIMD-optimized find for integer types
		/// \tparam T Unsigned integer type to search for
		/// \tparam config Algorithm configuration (default: algorithmFindConfig)
		/// \param data[in]: Pointer to array of elements to search
		/// \param n[in]: Number of elements in the array
		/// \param val[in]: Value to find in the array
		/// \return Position of the first element == val or n if not found
		template<typename T,
				 const AlgorithmFindConfig &config=algorithmFindConfig>
#if __cplusplus > 201709L
			requires std::unsigned_integral<T>
#endif
		constexpr size_t find_uXX_simd(const T *data,
									   const size_t n,
									   const T val) noexcept {
			using S = SIMDSelector<T>;

			const auto t = S::set1(val);
			size_t i = 0;
			for (; (i+S::LIMBS) <= n; i+=S::LIMBS) {
				const auto d = S::template load<config.aligned_instructions>(data + i);
				const auto s = d == t;
				if (s) [[unlikely]] {
					return i + ffs<T>(s) - 1u;
				}
			}

			for (; i < n; i++) {
				if (data[i] == val) {
					return i;
				}
			}

			return i;
		}
	} // end namespace internal

	/// Finds the first occurrence of a value in a range (sequential version)
	/// \tparam InputIt Forward iterator type for the range
	/// \tparam config Algorithm configuration (default: algorithmFindConfig)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param value[in]: Value to find in the range
	/// \return Iterator to the first occurrence of value, or last if not found
	template<class InputIt,
			 const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt>
#endif
	constexpr InputIt find(InputIt first,
						   InputIt last,
						   const typename InputIt::value_type& value) noexcept {

		using T = InputIt::value_type;
		if constexpr (config.assume_sorted) {
			if constexpr (config.use_interpolation_search) {
				cryptanalysislib::search::binary_search(first, last, value);
			} else {
				cryptanalysislib::search::binary_search(first, last, value);
			}
		}

		if constexpr (std::is_unsigned_v<T>) {
			const size_t t = internal::find_uXX_simd<T, config>(&(*first),
											static_cast<size_t>(std::distance(first, last)),
											value);
            std::advance(first, t);
            return first;
		}

		for (; first != last; ++first) {
			if (*first == value) {
				return first;
			}
		}

		return last;
	}

	/// Finds the first element satisfying a predicate (sequential version)
	/// \tparam InputIt Forward iterator type for the range
	/// \tparam UnaryPred Predicate type to test elements
	/// \tparam config Algorithm configuration (default: algorithmFindConfig)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param p[in]: Unary predicate function
	/// \return Iterator to the first element satisfying the predicate, or last if none found
	template<class InputIt,
			 class UnaryPred,
			 const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
    			 std::regular_invocable<UnaryPred,
									const typename InputIt::value_type&>
#endif
	constexpr InputIt find_if(InputIt first,
							  InputIt last,
							  UnaryPred p) noexcept {
		for (; first != last; ++first) {
			if (p(*first)) {
				return first;
			}
		}

		return last;
	}

	/// Finds the first element not satisfying a predicate (sequential version)
	/// \tparam InputIt Forward iterator type for the range
	/// \tparam UnaryPred Predicate type to test elements
	/// \tparam config Algorithm configuration (default: algorithmFindConfig)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param q[in]: Unary predicate function
	/// \return Iterator to the first element not satisfying the predicate, or last if none found
	template<class InputIt,
			 class UnaryPred,
			 const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
    			 std::regular_invocable<UnaryPred,
									const typename InputIt::value_type&>
#endif
	constexpr InputIt find_if_not(InputIt first,
								  InputIt last,
								  UnaryPred q) noexcept {
	    for (; first != last; ++first) {
		    if (!q(*first)) {
		    	return first;
		    }
	    }

	    return last;
	}

	/// Finds the first occurrence of a value in a range (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam config Algorithm configuration (default: algorithmFindConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param value[in]: Value to find in the range
	/// \return Iterator to the first occurrence of value, or last if not found
	template <class ExecPolicy,
			  class RandIt,
			  const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
	requires std::random_access_iterator<RandIt>
#endif
	RandIt find(ExecPolicy &&policy,
				RandIt first,
				RandIt last,
				const typename RandIt::value_type& value) noexcept {
		using diff_t = typename std::iterator_traits<RandIt>::difference_type;
		const diff_t size = std::distance(first, last);
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::find<RandIt, config>(first, last, value);
		}

		std::atomic<diff_t> extremum(size);

		internal::parallel_chunk_for_1_wait(std::forward<ExecPolicy>(policy), first, last,
			[&first, &extremum, &value](RandIt chunk_first,
									  RandIt chunk_last)
									  __attribute__((always_inline)) {
				if (std::distance(first, chunk_first) > extremum) {
					// already found by another task
					return;
				}

				RandIt chunk_res = cryptanalysislib::find
                                      <RandIt, config>
                                      (chunk_first, chunk_last, value);
				if (chunk_res != chunk_last) {
					// Found, update exremum using a priority update CAS, as discussed in
					// "Reducing Contention Through Priority Updates", PPoPP '13
					const diff_t k = std::distance(first, chunk_res);
					for (diff_t old = extremum; k < old; old = extremum) {
						extremum.compare_exchange_weak(old, k);
					}
				}
			}, (void*)nullptr,
			8,
			nthreads);
		// use small tasks so later ones may exit early if item is already found
		return extremum == size ? last : first + extremum;
	}

	/// Finds the first element satisfying a predicate (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam UnaryPred Predicate type to test elements
	/// \tparam config Algorithm configuration (default: algorithmFindConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param p[in]: Unary predicate function
	/// \return Iterator to the first element satisfying the predicate, or last if none found
	template <class ExecPolicy,
			  class RandIt,
	          class UnaryPred,
			  const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
	requires std::random_access_iterator<RandIt> &&
			 std::regular_invocable<UnaryPred,
									const typename RandIt::value_type&>
#endif
	RandIt find_if(ExecPolicy &&policy,
				   RandIt first,
				   RandIt last,
				   UnaryPred p) noexcept {
		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::find_if<RandIt, decltype(p), config>(first, last, p);
		}

		using diff_t = typename std::iterator_traits<RandIt>::difference_type;
		std::atomic<diff_t> extremum(size);

		internal::parallel_chunk_for_1_wait(std::forward<ExecPolicy>(policy), first, last,
			[&first, &extremum, &p](RandIt chunk_first,
										  RandIt chunk_last)
										  __attribute__((always_inline)) {
				if (std::distance(first, chunk_first) > extremum) {
					// already found by another task
					return;
				}

				RandIt chunk_res = cryptanalysislib::find_if
					<RandIt, UnaryPred, config>
					(chunk_first, chunk_last, p);
				if (chunk_res != chunk_last) {
					// Found, update exremum using a priority update CAS, as discussed in
					// "Reducing Contention Through Priority Updates", PPoPP '13
					const diff_t k = std::distance(first, chunk_res);
					for (diff_t old = extremum; k < old; old = extremum) {
						extremum.compare_exchange_weak(old, k);
					}
				}
			}, (void*)nullptr,
			8,
			nthreads);

		// use small tasks so later ones may exit early if item is already found
		return (size_t)extremum == size ? last : first + extremum;
	}

	/// Finds the first element not satisfying a predicate (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam UnaryPredicate Predicate type to test elements
	/// \tparam config Algorithm configuration (default: algorithmFindConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param p[in]: Unary predicate function
	/// \return Iterator to the first element not satisfying the predicate, or last if none found
	template <class ExecPolicy,
			  class RandIt,
			  class UnaryPredicate,
			  const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	RandIt find_if_not(ExecPolicy &&policy,
					   RandIt first,
					   RandIt last,
					   UnaryPredicate p) noexcept {
		return cryptanalysislib::find_if
				<ExecPolicy, RandIt, decltype(std::not_fn(p)), config>
				(std::forward<ExecPolicy>(policy), first, last,
			std::not_fn(p)
		);
	}

    /// Finds the last subsequence in a range that matches another range (default comparison)
    /// \tparam ForwardIt1 Forward iterator type for the main range
    /// \tparam ForwardIt2 Forward iterator type for the subsequence range
    /// \param first[in]: Iterator to the beginning of the main range
    /// \param last[in]: Iterator to the end of the main range
    /// \param s_first[in]: Iterator to the beginning of the subsequence
    /// \param s_last[in]: Iterator to the end of the subsequence
    /// \return Iterator to the beginning of the last matching subsequence, or last if not found
    template<class ForwardIt1, 
             class ForwardIt2>
#if __cplusplus > 201709L
		requires std::forward_iterator<ForwardIt1> &&
                 std::forward_iterator<ForwardIt2>
#endif
    constexpr
    ForwardIt1 find_end(ForwardIt1 first,
                        ForwardIt1 last,
                        ForwardIt2 s_first,
                        ForwardIt2 s_last) noexcept {
        if (s_first == s_last) {
            return last;
        }
     
        ForwardIt1 result = last;
        while (true) {
            ForwardIt1 new_result = std::search(first, last, s_first, s_last);
            if (new_result == last) {
                break;
            } else {
                result = new_result;
                first = result;
                ++first;
            }
        }

        return result;
    }

    /// Finds the last subsequence in a range that matches another range (custom predicate)
    /// \tparam ForwardIt1 Forward iterator type for the main range
    /// \tparam ForwardIt2 Forward iterator type for the subsequence range
    /// \tparam BinaryPred Binary predicate type for element comparison
    /// \param first[in]: Iterator to the beginning of the main range
    /// \param last[in]: Iterator to the end of the main range
    /// \param s_first[in]: Iterator to the beginning of the subsequence
    /// \param s_last[in]: Iterator to the end of the subsequence
    /// \param p[in]: Binary predicate for element comparison
    /// \return Iterator to the beginning of the last matching subsequence, or last if not found
    template<class ForwardIt1, 
             class ForwardIt2, 
             class BinaryPred>
    #if __cplusplus > 201709L
    		requires std::forward_iterator<ForwardIt1> &&
                     std::forward_iterator<ForwardIt2> && 
    			     std::regular_invocable<BinaryPred, bool>
    #endif
    constexpr //< since C++20
    ForwardIt1 find_end(ForwardIt1 first,
                        ForwardIt1 last,
                        ForwardIt2 s_first,
                        ForwardIt2 s_last,
                        BinaryPred p) {
        if (s_first == s_last) {
            return last;
        }
     
        ForwardIt1 result = last;
        while (true) {
            ForwardIt1 new_result = cryptanalysislib::search(first, last, s_first, s_last, p);
            if (new_result == last) {
                break;
            } else {
                result = new_result;
                first = result;
                ++first;
            }
        }

        return result;
    }

    /// TODO doc
    template<class InputIt,
             class ForwardIt>
    InputIt find_first_of(InputIt first,
                          InputIt last,
                          const ForwardIt s_first,
                          const ForwardIt s_last) {
        for (; first != last; ++first)
            for (ForwardIt it = s_first; it != s_last; ++it)
                if (*first == *it)
                    return first;
        return last;
    }

    /// TODO doc
    template<class InputIt,
             class ForwardIt,
             class BinaryPred>
    InputIt find_first_of(InputIt first,
                          InputIt last,
                          const ForwardIt s_first, 
                          const ForwardIt s_last,
                          BinaryPred p) {
        for (; first != last; ++first)
            for (ForwardIt it = s_first; it != s_last; ++it)
                if (p(*first, *it))
                    return first;
        return last;
    }
    
    /// TODO doc
    template<class ForwardIt>
    ForwardIt adjacent_find(ForwardIt first,
                            ForwardIt last) {
        if (first == last)
            return last;
     
        ForwardIt next = first;
        ++next;
     
        for (; next != last; ++next, ++first)
            if (*first == *next)
                return first;
     
        return last;
    }
    
    /// TODO doc
    template<class ForwardIt,
             class BinaryPred>
    ForwardIt adjacent_find(ForwardIt first,
                            ForwardIt last,
                            BinaryPred p) {
        if (first == last)
            return last;
     
        ForwardIt next = first;
        ++next;
     
        for (; next != last; ++next, ++first)
            if (p(*first, *next))
                return first;
     
        return last;
    }
} // end namespace cryptanalysislib
#endif //FIND_H
