#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_H

#include <future>
#include <iterator>
#include <utility>
#include <vector>

#include "algorithm/algorithm.h"
#include "algorithm/numeric/transform_inclusive_scan.h"

namespace cryptanalysislib {
	/// Writes `init, b(init, u(x[0])), b(b(init, u(x[0])), u(x[1])), ...` to
	/// `d_first`, i.e. the i-th output excludes the i-th input (like
	/// `std::transform_exclusive_scan`, sequential: the operations are applied
	/// in this order). `d_first == first` (inplace) is allowed.
	/// \param first[in]: beginning of the input range
	/// \param last[in]: end of the input range
	/// \param d_first[out]: beginning of the output range
	/// \param init[in]: first output, initial value of the accumulator
	/// \param binary_op[in]: "sum", `binary_op(accumulator, unary_op(x))`
	/// \param unary_op[in]: transformation applied to every element
	/// \return iterator past the last written element
	template<class InputIt,
	         class OutputIt,
	         class T,
	         class BinaryOp,
	         class UnaryOp>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt transform_exclusive_scan(InputIt first,
	                                            InputIt last,
	                                            OutputIt d_first,
	                                            T init,
	                                            BinaryOp binary_op,
	                                            UnaryOp unary_op) noexcept {
		for (; first != last; ++first, (void) ++d_first) {
			// read before the write: inplace works
			T next = binary_op(init, unary_op(*first));
			*d_first = std::move(init);
			init = std::move(next);
		}
		return d_first;
	}

	/// Writes `init, b(init, u(x[0])), ...` to `d_first` (parallel version of
	/// `transform_exclusive_scan`): the reductions of the chunks are computed
	/// in parallel (`internal::transform_scan_chunk_sums`), combined
	/// sequentially into the carry of every chunk, and every chunk is
	/// scanned in parallel starting with its carry.
	/// NOTE: `binary_op` must be associative, the ranges must not overlap
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the input range
	/// \param last[in]: end of the input range
	/// \param d_first[out]: beginning of the output range
	/// \param init[in]: first output, initial value of the accumulator
	/// \param binary_op[in]: "sum", `binary_op(accumulator, unary_op(x))`
	/// \param unary_op[in]: transformation applied to every element
	/// \return iterator past the last written element
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class T,
	         class BinaryOp,
	         class UnaryOp,
	         const AlgorithmTransformScanConfig &config = algorithmTransformScanConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 transform_exclusive_scan(ExecPolicy &&policy,
	                                 RandIt1 first,
	                                 RandIt1 last,
	                                 RandIt2 d_first,
	                                 T init,
	                                 BinaryOp binary_op,
	                                 UnaryOp unary_op) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::transform_exclusive_scan(first, last, d_first, init, binary_op, unary_op);
		}

		const std::vector<T> sums = internal::transform_scan_chunk_sums<T>(policy, first, n, nthreads,
		                                                                   binary_op, unary_op);
		auto &task_pool = *policy.pool();
		std::vector<std::future<void>> futures;
		T carry = std::move(init);
		for (uint32_t k = 0; k < nthreads; ++k) {
			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			futures.emplace_back(task_pool.submit([&binary_op, &unary_op, first, d_first, lo, hi, carry]() {
				cryptanalysislib::transform_exclusive_scan(first + lo, first + hi, d_first + lo,
				                                           carry, binary_op, unary_op);
			}));
			if (k + 1 < nthreads) {
				carry = binary_op(std::move(carry), sums[k]);
			}
		}
		internal::get_futures(futures);
		return d_first + n;
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_TRANSFORM_EXCLUSIVE_SCAN_H
