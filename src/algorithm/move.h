#ifndef CRYPTANALYSISLIB_ALGORITHM_MOVE_H
#define CRYPTANALYSISLIB_ALGORITHM_MOVE_H

#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "memory/memory.h"
#include "algorithm/algorithm.h"

/// NOTE: the requires-clauses make these overloads more constrained than
/// 	`std::move(first, last, d_first)`/`std::move_backward`, so unqualified
/// 	calls which see both (`using namespace cryptanalysislib;` + ADL on
/// 	`std::` iterators) are not ambiguous.
namespace cryptanalysislib {
	/// Moves the elements of [first, last) to the range starting at
	/// `d_first`, starting with the first element (like `std::move`, the
	/// algorithm). `d_first` must not be in [first, last), moving to the
	/// left within the same range is fine.
	/// \tparam InputIt input iterator type of the source range
	/// \tparam OutputIt output iterator type of the destination range
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the destination range
	/// \return iterator past the last moved-to element
	template<class InputIt,
	         class OutputIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt move(InputIt first,
	                        InputIt last,
	                        OutputIt d_first) noexcept {
		using T = std::iter_value_t<InputIt>;
		// NOTE: nested, `std::iter_value_t` of e.g. a `std::back_insert_iterator`
		// 	is ill-formed
		if constexpr (std::contiguous_iterator<InputIt> &&
		              std::contiguous_iterator<OutputIt>) {
			if constexpr (std::is_same_v<T, std::iter_value_t<OutputIt>> &&
			              std::is_trivially_copyable_v<T>) {
				const size_t n = static_cast<size_t>(last - first);
				if (n == 0) {
					return d_first;
				}
				// element count, the ranges may overlap
				cryptanalysislib::memmove(std::to_address(d_first), std::to_address(first), n);
				return d_first + n;
			}
		}

		for (; first != last; ++first, ++d_first) {
			*d_first = std::move(*first);
		}
		return d_first;
	}

	/// Moves [first, last) to the range ending at `d_last`, starting with
	/// the last element (like `std::move_backward`). `d_last` must not be in
	/// (first, last], moving to the right within the same range is fine.
	/// \tparam BidirIt1 bidirectional iterator type of the source range
	/// \tparam BidirIt2 bidirectional iterator type of the destination range
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_last[out]: end of the destination range
	/// \return iterator to the first moved-to element
	template<class BidirIt1,
	         class BidirIt2>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<BidirIt1> &&
	             std::bidirectional_iterator<BidirIt2>
#endif
	constexpr BidirIt2 move_backward(BidirIt1 first,
	                                 BidirIt1 last,
	                                 BidirIt2 d_last) noexcept {
		using T = std::iter_value_t<BidirIt1>;
		if constexpr (std::contiguous_iterator<BidirIt1> &&
		              std::contiguous_iterator<BidirIt2> &&
		              std::is_same_v<T, std::iter_value_t<BidirIt2>> &&
		              std::is_trivially_copyable_v<T>) {
			const size_t n = static_cast<size_t>(last - first);
			if (n == 0) {
				return d_last;
			}
			// element count, the ranges may overlap
			cryptanalysislib::memmove(std::to_address(d_last - n), std::to_address(first), n);
			return d_last - n;
		} else {
			while (first != last) {
				*(--d_last) = std::move(*(--last));
			}
			return d_last;
		}
	}

	/// Configuration for the parallel `move`
	struct AlgorithmMoveConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 16u;
	};
	constexpr static AlgorithmMoveConfig algorithmMoveConfig;

	/// Moves the elements of [first, last) to the range starting at
	/// `d_first` (parallel version of `move`): every thread moves one chunk.
	/// NOTE: unlike the sequential version the ranges must not overlap
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt1 random access iterator type of the source range
	/// \tparam RandIt2 random access iterator type of the destination range
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the destination range
	/// \return iterator past the last moved-to element
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmMoveConfig &config = algorithmMoveConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 move(ExecPolicy &&policy,
	             RandIt1 first,
	             RandIt1 last,
	             RandIt2 d_first) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::move(first, last, d_first);
		}

		auto futures = internal::parallel_chunk_for_2(std::forward<ExecPolicy>(policy),
		                                              first, last, d_first,
		                                              [](RandIt1 f, RandIt1 l, RandIt2 d) {
			                                              return cryptanalysislib::move(f, l, d);
		                                              },
		                                              (RandIt2 *)nullptr, nthreads);
		internal::get_futures(futures);
		return d_first + n;
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_MOVE_H
