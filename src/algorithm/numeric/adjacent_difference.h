#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_ADJACENT_DIFFERENCE_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_ADJACENT_DIFFERENCE_H

#include <functional>
#include <future>
#include <iterator>
#include <utility>
#include <vector>

#include "algorithm/algorithm.h"

namespace cryptanalysislib {
	/// Writes `x[0], op(x[1], x[0]), op(x[2], x[1]), ...` to `d_first` (like
	/// `std::adjacent_difference`). `d_first == first` (inplace) is allowed.
	/// \param first[in]: beginning of the input range
	/// \param last[in]: end of the input range
	/// \param d_first[out]: beginning of the output range
	/// \param op[in]: "difference", `op(current, previous)`
	/// \return iterator past the last written element
	template<class InputIt,
	         class OutputIt,
	         class BinaryOp>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt adjacent_difference(InputIt first,
	                                       InputIt last,
	                                       OutputIt d_first,
	                                       BinaryOp op) noexcept {
		if (first == last) {
			return d_first;
		}

		using T = std::iter_value_t<InputIt>;
		T prev = *first;
		*d_first = prev;
		while (++first != last) {
			// read before the write: inplace works
			T cur = *first;
			*++d_first = op(cur, std::move(prev));
			prev = std::move(cur);
		}
		return ++d_first;
	}

	/// same as above, with `-`
	template<class InputIt,
	         class OutputIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt adjacent_difference(InputIt first,
	                                       InputIt last,
	                                       OutputIt d_first) noexcept {
		return cryptanalysislib::adjacent_difference(first, last, d_first, std::minus<>());
	}

	/// Configuration for the parallel `adjacent_difference`
	struct AlgorithmAdjacentDifferenceConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 16u;
	};
	constexpr static AlgorithmAdjacentDifferenceConfig algorithmAdjacentDifferenceConfig;

	/// Writes `x[0], op(x[1], x[0]), op(x[2], x[1]), ...` to `d_first`
	/// (parallel version of `adjacent_difference`): every thread computes one
	/// chunk, reading the element before it.
	/// NOTE: unlike the sequential version the ranges must not overlap
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the input range
	/// \param last[in]: end of the input range
	/// \param d_first[out]: beginning of the output range
	/// \param op[in]: "difference", `op(current, previous)`
	/// \return iterator past the last written element
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class BinaryOp,
	         const AlgorithmAdjacentDifferenceConfig &config = algorithmAdjacentDifferenceConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 adjacent_difference(ExecPolicy &&policy,
	                            RandIt1 first,
	                            RandIt1 last,
	                            RandIt2 d_first,
	                            BinaryOp op) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::adjacent_difference(first, last, d_first, op);
		}

		auto &task_pool = *policy.pool();
		std::vector<std::future<void>> futures;
		for (uint32_t k = 0; k < nthreads; ++k) {
			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			futures.emplace_back(task_pool.submit([&op, first, d_first, lo, hi]() {
				if (lo == 0) {
					cryptanalysislib::adjacent_difference(first, first + hi, d_first, op);
					return;
				}
				for (size_t i = lo; i < hi; ++i) {
					d_first[i] = op(first[i], first[i - 1]);
				}
			}));
		}
		internal::get_futures(futures);
		return d_first + n;
	}

	/// same as above, with `-`
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmAdjacentDifferenceConfig &config = algorithmAdjacentDifferenceConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 adjacent_difference(ExecPolicy &&policy,
	                            RandIt1 first,
	                            RandIt1 last,
	                            RandIt2 d_first) noexcept {
		return cryptanalysislib::adjacent_difference
			<ExecPolicy, RandIt1, RandIt2, std::minus<>, config>
			(std::forward<ExecPolicy>(policy), first, last, d_first, std::minus<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_ADJACENT_DIFFERENCE_H
