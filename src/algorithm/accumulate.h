#ifndef CRYPTANALYSISLIB_ALGORITHM_ACCUMULATE_H
#define CRYPTANALYSISLIB_ALGORITHM_ACCUMULATE_H

#include "thread/thread.h"
#include "algorithm/algorithm.h"
#include "simd/simd.h"

#include <numeric>

namespace cryptanalysislib {
	struct AlgorithmAccumulateConfig : public AlgorithmConfig {
	    constexpr static size_t min_size_per_thread = 1u<<10u;
	};
	constexpr static AlgorithmAccumulateConfig algorithmAccumulateConfig;

	namespace internal {
		template<typename T>
		constexpr T accumulate_simd_int_plus(const T *data,
							  const size_t n,
							  const T init) noexcept {
#ifdef USE_AVX512F
			constexpr static uint32_t limbs = 64/sizeof(T);
#else
			constexpr static uint32_t limbs = 32/sizeof(T);
#endif
			using S = TxN_t<T, limbs>;
			T ret = init;

			S acc = S::set1(0);
			size_t i = 0;
			for (; (i+limbs) <= n; i+=limbs) {
				const auto d = S::load(data + i);
				acc = acc + d;
			}

			for (uint32_t j = 0; j < limbs; j++) {
				ret += acc[j];
			}

			//
			for (; i < n; i++) {
				ret += data[i];
			}
			return ret;
		}
	} // end namespace internal

	///
	/// @tparam InputIt
	/// @tparam T
	/// @param first
	/// @param last
	/// @param init
	/// @return
	template<class InputIt,
			 const AlgorithmAccumulateConfig &config=algorithmAccumulateConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<InputIt>
#endif
	constexpr InputIt::value_type accumulate(InputIt first,
						   InputIt last,
						   typename  InputIt::value_type init) noexcept {
		for (; first != last; ++first) {
			init = std::move(init) + *first;
		}

		return init;
	}

	///
	/// @tparam InputIt
	/// @tparam BinaryOperation
	/// @param first
	/// @param last
	/// @param init
	/// @param op
	/// @return
	template<class InputIt,
			 class BinaryOperation,
			 const AlgorithmAccumulateConfig &config=algorithmAccumulateConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<InputIt>
#endif
	constexpr InputIt::value_type accumulate(InputIt first,
						   InputIt last,
						   typename InputIt::value_type init,
						   BinaryOperation op) {
		for (; first != last; ++first) {
			init = op(std::move(init), *first);
		}
		return init;
	}



	template <class ExecPolicy,
			  class RandIt,
			  const AlgorithmAccumulateConfig &config=algorithmAccumulateConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	typename std::iterator_traits<RandIt>::value_type
	accumulate(ExecPolicy&& policy,
			 RandIt first,
			 RandIt last,
			 typename RandIt::value_type init) noexcept {

		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::accumulate<RandIt, config>(first, last, init);
		}

		using D = RandIt::difference_type;
		using T = RandIt::value_type;
		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first, last,
			cryptanalysislib::accumulate<RandIt, config>,
			(T *)0,
			1, nthreads, T{});
		return init + std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), (T)0, std::plus<T>());
	}
} // end namespace


#endif
