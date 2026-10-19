#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_IOTA_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_IOTA_H

#include <future>
#include <iterator>
#include <vector>

#include "algorithm/algorithm.h"

namespace cryptanalysislib {
	/// Fills [first, last) with `value`, `value + 1`, ... (like `std::iota`,
	/// the value is incremented with `++`)
	/// NOTE: the requires-clause makes it more constrained than `std::iota`,
	/// 	so unqualified calls which see both are not ambiguous.
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param value[in]: first value
	template<class ForwardIt,
	         class T>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr void iota(ForwardIt first,
	                    ForwardIt last,
	                    T value) noexcept {
		for (; first != last; ++first, (void) ++value) {
			*first = value;
		}
	}

	/// Configuration for the parallel `iota`
	struct AlgorithmIotaConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 16u;
	};
	constexpr static AlgorithmIotaConfig algorithmIotaConfig;

	/// Fills [first, last) with `value`, `value + 1`, ... (parallel version
	/// of `iota`, there is no `std::` one): the chunk [lo, hi) starts with
	/// `T(value + lo)` and increments it with `++`.
	/// NOTE: `T(value + i)` must equal `i` times `++value`, which holds for
	/// 	integers (incl. the wrap around of unsigned types) and iterators,
	/// 	but for floating point values only as long as they are exact.
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param value[in]: first value
	template<class ExecPolicy,
	         class RandIt,
	         class T,
	         const AlgorithmIotaConfig &config = algorithmIotaConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt> &&
	             requires(const T &v, const std::iter_difference_t<RandIt> d) { static_cast<T>(v + d); }
#endif
	void iota(ExecPolicy &&policy,
	          RandIt first,
	          RandIt last,
	          T value) noexcept {
		using D = std::iter_difference_t<RandIt>;
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			cryptanalysislib::iota(first, last, value);
			return;
		}

		auto &task_pool = *policy.pool();
		std::vector<std::future<void>> futures;
		for (uint32_t k = 0; k < nthreads; ++k) {
			const D lo = static_cast<D>((n * k) / nthreads), hi = static_cast<D>((n * (k + 1)) / nthreads);
			futures.emplace_back(task_pool.submit([first, lo, hi, value]() {
				cryptanalysislib::iota(first + lo, first + hi, static_cast<T>(value + lo));
			}));
		}
		internal::get_futures(futures);
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_IOTA_H
