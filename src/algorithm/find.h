#ifndef CRYPTANALYSISLIB_ALGORITHM_FIND_H
#define CRYPTANALYSISLIB_ALGORITHM_FIND_H

#include "thread/thread.h"
#include "algorithm/algorithm.h"
#include "algorithm/bits/ffs.h"
#include "simd/simd.h"

namespace cryptanalysislib {
	struct AlgorithmFindConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u<<10u;
	};
	constexpr static AlgorithmFindConfig algorithmFindConfig;

	namespace internal {

		/// \tparam T
		/// \param data
		/// \param n
		/// \param val
		/// \return the position of the first element == val or n
		template<typename T,
				 const AlgorithmFindConfig &config=algorithmFindConfig>
#if __cplusplus > 201709L
    requires std::unsigned_integral<T>
#endif
		constexpr size_t find_uXX_simd(const T *data,
										const size_t n,
										const T val) noexcept {
#ifdef USE_AVX512F
			constexpr static uint32_t limbs = 64/sizeof(T);
#else
			constexpr static uint32_t limbs = 32/sizeof(T);
#endif
			using S = TxN_t<T, limbs>;

			const auto t = S::set1(val);
			size_t i = 0;
			for (; (i+limbs) <= n; i+=limbs) {
				const auto d = S::load(data + i);
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
	}// end namespace internal

	/// \tparam InputIt
	/// \tparam config
	/// \param first
	/// \param last
	/// \param value
	/// \return
	template<class InputIt,
			 const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
	requires std::bidirectional_iterator<InputIt>
#endif
	constexpr InputIt find(InputIt first,
						   InputIt last,
						   const typename InputIt::value_type& value) noexcept {

		using T = InputIt::value_type;
		if constexpr (std::is_unsigned_v<T>) {
			return internal::find_uXX_simd(&(*first),
											static_cast<size_t>(std::distance(first, last)),
											value);
		}

		for (; first != last; ++first) {
			if (*first == value) {
				return first;
			}
		}

		return last;
	}

	/// \tparam InputIt
	/// \tparam UnaryPred
	/// \param first
	/// \param last
	/// \param p
	/// \return
	template<class InputIt,
			 class UnaryPred>
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

	/// \tparam InputIt
	/// \tparam UnaryPred
	/// \param first
	/// \param last
	/// \param q
	/// \return
	template<class InputIt,
			 class UnaryPred>
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
		const size_t size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::find<RandIt, config>(first, last, value);
		}

		using diff_t = typename std::iterator_traits<RandIt>::difference_type;
		std::atomic<diff_t> extremum(size);

		internal::parallel_chunk_for_1_wait(std::forward<ExecPolicy>(policy), first, last,
			[&first, &extremum, &value](RandIt chunk_first,
									  RandIt chunk_last)
									  __attribute__((always_inline)) {
				if (std::distance(first, chunk_first) > extremum) {
					// already found by another task
					return;
				}

				RandIt chunk_res = cryptanalysislib::find<RandIt, config>(chunk_first, chunk_last, value);
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

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam UnaryPredicate
	/// \param policy
	/// \param first
	/// \param last
	/// \param p
	/// \return
	template <class ExecPolicy,
			  class RandIt,
	          class UnaryPredicate,
			 const AlgorithmFindConfig &config = algorithmFindConfig>
#if __cplusplus > 201709L
	requires std::random_access_iterator<RandIt>
#endif
	RandIt find_if(ExecPolicy &&policy,
				   RandIt first,
				   RandIt last,
				   UnaryPredicate p) noexcept {
		const size_t size = static_cast<size_t>(std::distance(first, last));
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

				RandIt chunk_res = cryptanalysislib::find_if(chunk_first, chunk_last, p);
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

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam UnaryPredicate
	/// \param policy
	/// \param first
	/// \param last
	/// \param p
	/// \return
	template <class ExecPolicy,
			  class RandIt,
			  class UnaryPredicate>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	RandIt find_if_not(ExecPolicy &&policy,
					   RandIt first,
					   RandIt last,
					   UnaryPredicate p) noexcept {
		return std::find_if(std::forward<ExecPolicy>(policy), first, last,
			std::not_fn(p)
		);
	}
}
#endif //FIND_H
