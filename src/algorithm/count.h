#ifndef CRYPTANALYSISLIB_ALGORITHM_COUNT_H
#define CRYPTANALYSISLIB_ALGORITHM_COUNT_H

#include <numeric>

#include "thread/thread.h"
#include "algorithm/algorithm.h"
#include "simd/simd.h"

namespace cryptanalysislib {

	struct AlgorithmCountConfig : public AlgorithmConfig {
	    const size_t min_size_per_thread = 1u<<10u;
	    const bool aligned_instructions = false;
	};
	constexpr static AlgorithmCountConfig algorithmCountConfig;

	/// \tparam RandIt
	/// \tparam UnaryPredicate
	/// \tparam config
	/// \param first
	/// \param last
	/// \param p
	/// \return
	template <class Iterator,
			  class UnaryPredicate,
			  const AlgorithmCountConfig &config=algorithmCountConfig>
#if __cplusplus > 201709L
    requires std::forward_iterator<Iterator> &&
    		 std::regular_invocable<UnaryPredicate,
									const typename Iterator::value_type&>
#endif
	typename std::iterator_traits<Iterator>::difference_type
	count_if(Iterator first,
			 Iterator last,
			 UnaryPredicate p) noexcept {
		typename std::iterator_traits<Iterator>::difference_type ret = 0;
		for (; first != last; ++first) {
			if (p(*first)) {
				++ret;
			}
		}

		return ret;
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam UnaryPredicate
	/// \tparam config
	/// \param policy
	/// \param first
	/// \param last
	/// \param p
	/// \return
	template <class ExecPolicy,
			  class RandIt,
			  class UnaryPredicate,
			  const AlgorithmCountConfig &config=algorithmCountConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt> &&
    		 std::regular_invocable<UnaryPredicate,
									const typename RandIt::value_type&>
#endif
	typename std::iterator_traits<RandIt>::difference_type
	count_if(ExecPolicy&& policy,
			 RandIt first,
			 RandIt last,
			 UnaryPredicate p) noexcept {

		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::count_if<RandIt, decltype(p), config>(first, last, p);
		}

		using T = RandIt::difference_type;
		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first, last,
			cryptanalysislib::count_if<RandIt, UnaryPredicate, config>,
			(T*)nullptr,
			1, nthreads, p);

		return std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), (T)0, std::plus<T>());
	}

	namespace internal {

		/// \tparam T
		/// \param data
		/// \param n
		/// \param val
		/// \return
		template<typename T,
			     const AlgorithmCountConfig &config=algorithmCountConfig>
#if __cplusplus > 201709L
    requires std::unsigned_integral<T>
#endif
		constexpr size_t count_uXX_simd(const T *data,
										const size_t n,
										const T val) noexcept {
#ifdef USE_AVX512F
			constexpr uint32_t limbs = 64/sizeof(T);
#else
			constexpr uint32_t limbs = 32/sizeof(T);
#endif
			using S = TxN_t<T, limbs>;

			size_t ret = 0;
			const auto t = S::set1(val);
			size_t i = 0;
			for (; (i+limbs) <= n; i+=limbs) {
				const auto d = S::template load<config.aligned_instructions>(data + i);
				const auto s = d == t;
				if (s) {
					ret += popcount::popcount(s);
				}
			}

			// tailmngt
			for (;i<n;i++) {
				ret += (data[i] == val);
			}

			return ret;
		}
	}// end namespace internal

	/// \tparam RandIt
	/// \param first
	/// \param last
	/// \param value
	/// \return
	template <class RandIt,
			  const AlgorithmCountConfig &config=algorithmCountConfig>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<RandIt>
#endif
	RandIt::difference_type count(RandIt first,
								  RandIt last,
								  const typename RandIt::value_type & value) noexcept {
		using T = RandIt::value_type;
		if constexpr (std::is_unsigned_v<T>) {
			return internal::count_uXX_simd(&(*first),
											static_cast<size_t>(std::distance(first, last)),
											value);
		}

		auto p = [&value](const T& test) __attribute__((always_inline)) {
			return test == value;
		};

		return cryptanalysislib::count_if
			<RandIt, decltype(p), config>
			(first, last, p);
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \param policy
	/// \param first
	/// \param last
	/// \param value
	/// \return
	template <class ExecPolicy,
			  class RandIt,
			  const AlgorithmCountConfig &config=algorithmCountConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	typename std::iterator_traits<RandIt>::difference_type
	count(ExecPolicy&& policy,
	      RandIt first,
	      RandIt last,
	      const typename RandIt::value_type& value) noexcept {
		using T = typename RandIt::value_type;

		auto p = [&value](const T& test) __attribute__((always_inline)) {
			 return test == value;
		};
		return cryptanalysislib::count_if
			<ExecPolicy, RandIt, decltype(p), config>
			(std::forward<ExecPolicy>(policy),
			first, last, p);
	}

} // end namespace cryptanalysislib
#endif
