#ifndef CRYPTANALYSISLIB_ALGORITHM_ACCUMULATE_H
#define CRYPTANALYSISLIB_ALGORITHM_ACCUMULATE_H

#include <numeric>

#include "algorithm/algorithm.h"
#include "simd/simd.h"

namespace cryptanalysislib {
	/// Configuration for accumulate algorithm
	struct AlgorithmAccumulateConfig : public AlgorithmConfig {
        /// Minimum data size per thread
	    const size_t min_size_per_thread = 1u<<10u;
        /// Whether to use aligned memory access
	    const bool aligned_instructions = false;
	};
	constexpr static AlgorithmAccumulateConfig algorithmAccumulateConfig;

	namespace internal {

		/// SIMD accelerated implementation for accumulating numeric values
		/// \tparam T [in] Element type to accumulate
		/// \param data [in] Pointer to array of elements
		/// \param n [in] Number of elements to process
		/// \param init [in] Initial value for accumulation
		/// \return Accumulated sum of elements with initial value
		template<typename T,
				 const AlgorithmAccumulateConfig &config = algorithmAccumulateConfig>
		constexpr T accumulate_simd_int_plus(const T *data,
							                 const size_t n,
							                 const T init) noexcept {
			using S = SIMDSelector<T>;
			T ret = init;

			S acc = S::set1(0);
			size_t i = 0;
			for (; (i+S::LIMBS) <= n; i+=S::LIMBS) {
				const auto d = S::template load<config.aligned_instructions>(data + i);
				acc = acc + d;
			}

			for (uint32_t j = 0; j < S::LIMBS; j++) {
				ret += acc[j];
			}

			// Process remaining elements
			for (; i < n; i++) {
				ret += data[i];
			}
			return ret;
		}
	} // end namespace internal

	/// Sequential implementation of accumulate with addition operation
	/// \tparam InputIt [in] Input iterator type
	/// \param first [in] Iterator to first element
	/// \param last [in] Iterator past the last element
	/// \param init [in] Initial accumulation value
	/// \return Sum of all elements plus initial value
	template<class InputIt,
			 const AlgorithmAccumulateConfig &config=algorithmAccumulateConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<InputIt>
#endif
	constexpr InputIt::value_type accumulate(InputIt first,
											 const InputIt last,
											 typename InputIt::value_type init) noexcept {
        using T = typename std::iterator_traits<InputIt>::value_type;

		// For contiguous arrays of arithmetic types, use SIMD-accelerated implementation
		if constexpr (std::is_arithmetic_v<T>) {
		    const size_t n = std::distance(first, last);
		    return internal::accumulate_simd_int_plus(first, n, init);
		}
		
		// Generic implementation for all other cases
		for (; first != last; ++first) {
			init = std::move(init) + *first;
		}

		return init;
	}

	/// Sequential implementation of accumulate with custom binary operation
	/// \tparam InputIt [in] Input iterator type
	/// \tparam BinaryOperation [in] Binary operation type
	/// \param first [in] Iterator to first element
	/// \param last [in] Iterator past the last element
	/// \param init [in] Initial accumulation value
	/// \param op [in] Binary operation to apply
	/// \return Result of applying binary operation to all elements
	template<class InputIt,
			 class BinaryOperation,
			 const AlgorithmAccumulateConfig &config=algorithmAccumulateConfig>
#if __cplusplus > 201709L
    requires std::forward_iterator<InputIt> &&
    		 std::regular_invocable<BinaryOperation,
									const typename InputIt::value_type&,
									const typename InputIt::value_type&>
#endif
	constexpr InputIt::value_type accumulate(InputIt first,
						                     const InputIt last,
						                     const typename InputIt::value_type init,
						                     BinaryOperation op) {
		for (; first != last; ++first) {
			init = op(std::move(init), *first);
		}
		return init;
	}

	/// Parallel implementation of accumulate using execution policy
	/// \tparam ExecPolicy [in] Execution policy type
	/// \tparam RandIt[in]: Random access iterator type
	/// \tparam config[in]: Configuration for algorithm behavior
	/// \param policy[in]: Execution policy instance
	/// \param first[in]: Iterator to first element
	/// \param last[in]: Iterator past the last element
	/// \param init[in]: Initial accumulation value
	/// \return Sum of all elements plus initial value
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
