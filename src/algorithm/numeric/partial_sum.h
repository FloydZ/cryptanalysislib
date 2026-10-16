#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_PARTIAL_SUM_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_PARTIAL_SUM_H

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "algorithm/copy.h"
#include "algorithm/prefixsum.h"
#include "algorithm/numeric/transform_inclusive_scan.h"

namespace cryptanalysislib {
	/// Writes `x[0], op(x[0], x[1]), op(op(x[0], x[1]), x[2]), ...` to
	/// `d_first` (like `std::partial_sum`). The accumulator has the value
	/// type of `InputIt`. `d_first == first` (inplace) is allowed.
	/// \param first[in]: beginning of the input range
	/// \param last[in]: end of the input range
	/// \param d_first[out]: beginning of the output range
	/// \param op[in]: "sum", `op(accumulator, element)`
	/// \return iterator past the last written element
	template<class InputIt,
	         class OutputIt,
	         class BinaryOp>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt partial_sum(InputIt first,
	                               InputIt last,
	                               OutputIt d_first,
	                               BinaryOp op) noexcept {
		if (first == last) {
			return d_first;
		}

		std::iter_value_t<InputIt> acc = *first;
		*d_first = acc;
		while (++first != last) {
			acc = op(std::move(acc), *first);
			*++d_first = acc;
		}
		return ++d_first;
	}

	/// same as above, with `+`
	/// NOTE: contiguous `uint32_t` ranges use the inplace (SIMD)
	/// 	`algorithm::prefixsum` on the output
	template<class InputIt,
	         class OutputIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt partial_sum(InputIt first,
	                               InputIt last,
	                               OutputIt d_first) noexcept {
		// NOTE: nested, `std::iter_value_t` of e.g. a `std::back_insert_iterator`
		// 	is ill-formed
		if constexpr (std::contiguous_iterator<InputIt> &&
		              std::contiguous_iterator<OutputIt>) {
			if constexpr (std::is_same_v<std::iter_value_t<InputIt>, uint32_t> &&
			              std::is_same_v<std::iter_value_t<OutputIt>, uint32_t>) {
				if !consteval {
					const size_t n = static_cast<size_t>(last - first);
					if (n == 0) {
						return d_first;
					}

					uint32_t *out = std::to_address(d_first);
					if (out != std::to_address(first)) {
						cryptanalysislib::copy(first, last, d_first);
					}
					algorithm::prefixsum<uint32_t>(out, n);
					return d_first + n;
				}
			}
		}

		return cryptanalysislib::partial_sum(first, last, d_first, std::plus<>());
	}

	/// Writes `x[0], op(x[0], x[1]), op(op(x[0], x[1]), x[2]), ...` to
	/// `d_first` (parallel version of `partial_sum`, there is no `std::`
	/// one): the parallel `transform_inclusive_scan` with `std::identity`.
	/// The accumulator has the value type of `RandIt1`.
	/// NOTE: unlike the sequential version `op` must be associative and the
	/// 	ranges must not overlap
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the input range
	/// \param last[in]: end of the input range
	/// \param d_first[out]: beginning of the output range
	/// \param op[in]: "sum", `op(accumulator, element)`
	/// \return iterator past the last written element
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class BinaryOp,
	         const AlgorithmTransformScanConfig &config = algorithmTransformScanConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 partial_sum(ExecPolicy &&policy,
	                    RandIt1 first,
	                    RandIt1 last,
	                    RandIt2 d_first,
	                    BinaryOp op) noexcept {
		return cryptanalysislib::transform_inclusive_scan
			<ExecPolicy, RandIt1, RandIt2, BinaryOp, std::identity, config>
			(std::forward<ExecPolicy>(policy), first, last, d_first, op, std::identity());
	}

	/// same as above, with `+`
	/// NOTE: if it is not parallelised, the sequential version (incl. its
	/// 	SIMD path) is used
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmTransformScanConfig &config = algorithmTransformScanConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 partial_sum(ExecPolicy &&policy,
	                    RandIt1 first,
	                    RandIt1 last,
	                    RandIt2 d_first) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		if (is_seq<ExecPolicy>(policy) || (should_par(policy, config, n) < 2)) {
			return cryptanalysislib::partial_sum(first, last, d_first);
		}

		return cryptanalysislib::partial_sum
			<ExecPolicy, RandIt1, RandIt2, std::plus<>, config>
			(std::forward<ExecPolicy>(policy), first, last, d_first, std::plus<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_PARTIAL_SUM_H
