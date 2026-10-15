#ifndef CRYPTANALYSISLIB_ALGORITHM_UNIQUE_H
#define CRYPTANALYSISLIB_ALGORITHM_UNIQUE_H

#include <functional>
#include <future>
#include <iterator>
#include <utility>
#include <vector>

#include "algorithm/algorithm.h"

namespace cryptanalysislib {
	/// Removes all but the first element of every run of consecutive
	/// equivalent elements in [first, last) (like `std::unique`). The kept
	/// elements are moved to the front, the elements in [returned, last) are
	/// valid but unspecified.
	/// \tparam ForwardIt forward iterator type
	/// \tparam BinaryPred equivalence predicate type
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param p[in]: `p(a, b)` is true iff `a` and `b` are equivalent
	/// \return the new end of the range
	template<class ForwardIt,
	         class BinaryPred>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt unique(ForwardIt first,
	                           ForwardIt last,
	                           BinaryPred p) noexcept {
		if (first == last) {
			return last;
		}

		ForwardIt result = first;
		while (++first != last) {
			if (!p(*result, *first) && ++result != first) {
				*result = std::move(*first);
			}
		}
		return ++result;
	}

	/// Same as above, compares the elements with `==`
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \return the new end of the range
	template<class ForwardIt>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt unique(ForwardIt first,
	                           ForwardIt last) noexcept {
		return cryptanalysislib::unique(first, last, std::equal_to<>());
	}

	/// Copies [first, last) to `d_first`, but of every run of consecutive
	/// equivalent elements only the first one (like `std::unique_copy`)
	/// \tparam InputIt input iterator type of the source range
	/// \tparam OutputIt output iterator type of the destination range
	/// \tparam BinaryPred equivalence predicate type
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the destination range, must not
	/// 	overlap the source range
	/// \param p[in]: `p(a, b)` is true iff `a` and `b` are equivalent
	/// \return iterator past the last written element
	template<class InputIt,
	         class OutputIt,
	         class BinaryPred>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt unique_copy(InputIt first,
	                               InputIt last,
	                               OutputIt d_first,
	                               BinaryPred p) noexcept {
		if (first == last) {
			return d_first;
		}

		// the last written element: input iterators can not be read twice
		std::iter_value_t<InputIt> value = *first;
		*d_first = value;
		++d_first;
		while (++first != last) {
			if (!p(value, *first)) {
				value = *first;
				*d_first = value;
				++d_first;
			}
		}
		return d_first;
	}

	/// Same as above, compares the elements with `==`
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the destination range
	/// \return iterator past the last written element
	template<class InputIt,
	         class OutputIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt>
#endif
	constexpr OutputIt unique_copy(InputIt first,
	                               InputIt last,
	                               OutputIt d_first) noexcept {
		return cryptanalysislib::unique_copy(first, last, d_first, std::equal_to<>());
	}

	/// Configuration for the parallel `unique` and `unique_copy`
	struct AlgorithmUniqueConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 15u;
	};
	constexpr static AlgorithmUniqueConfig algorithmUniqueConfig;

	namespace internal {
		/// number of leading elements of the k-th of the `nthreads` chunks of
		/// the `n` elements at `first`, which are equivalent to the element
		/// before the chunk (0 for the first chunk). Only reads.
		template<class RandIt, class BinaryPred>
		size_t unique_skip(RandIt first,
		                   const size_t n,
		                   const uint32_t k,
		                   const uint32_t nthreads,
		                   BinaryPred &p) noexcept {
			if (k == 0) {
				return 0;
			}

			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			size_t i = lo;
			while ((i < hi) && p(first[lo - 1], first[i])) {
				++i;
			}
			return i - lo;
		}
	} // end namespace internal

	/// Removes all but the first element of every run of consecutive
	/// equivalent elements (parallel version of `unique`):
	/// 1. every chunk counts its leading elements which are equivalent to
	///    the element before it (read only),
	/// 2. every chunk without these leading elements is made unique inplace,
	/// 3. the chunks are moved together, in order.
	/// NOTE: `p` must be an equivalence relation (as for `std::unique`).
	/// NOTE: step 3 is sequential: a destination can overlap the source of an
	/// 	earlier chunk. Every move goes to the left, so a forward loop is safe.
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt random access iterator type
	/// \tparam BinaryPred equivalence predicate type
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param p[in]: `p(a, b)` is true iff `a` and `b` are equivalent
	/// \return the new end of the range
	template<class ExecPolicy,
	         class RandIt,
	         class BinaryPred,
	         const AlgorithmUniqueConfig &config = algorithmUniqueConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	RandIt unique(ExecPolicy &&policy,
	              RandIt first,
	              RandIt last,
	              BinaryPred p) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::unique(first, last, p);
		}

		auto &task_pool = *policy.pool();
		std::vector<size_t> skip(nthreads, 0);
		std::vector<std::future<void>> futures;
		for (uint32_t k = 1; k < nthreads; ++k) {
			futures.emplace_back(task_pool.submit([&skip, &p, first, n, k, nthreads]() {
				skip[k] = internal::unique_skip(first, n, k, nthreads, p);
			}));
		}
		internal::get_futures(futures);

		// [begin, end) of the kept elements of every chunk
		std::vector<std::future<std::pair<RandIt, RandIt>>> chunks;
		for (uint32_t k = 0; k < nthreads; ++k) {
			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			chunks.emplace_back(task_pool.submit([&p, b = first + lo + skip[k], e = first + hi]() {
				return std::pair<RandIt, RandIt>{b, cryptanalysislib::unique(b, e, p)};
			}));
		}

		RandIt out = first;
		for (auto &c : chunks) {
			auto [src, end] = c.get();
			if (out == src) {
				// nothing removed so far: no self move assignment
				out = end;
				continue;
			}
			for (; src != end; ++src, ++out) {
				*out = std::move(*src);
			}
		}
		return out;
	}

	/// same as above, compares the elements with `==`
	template<class ExecPolicy,
	         class RandIt,
	         const AlgorithmUniqueConfig &config = algorithmUniqueConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	RandIt unique(ExecPolicy &&policy,
	              RandIt first,
	              RandIt last) noexcept {
		return cryptanalysislib::unique
			<ExecPolicy, RandIt, std::equal_to<>, config>
			(std::forward<ExecPolicy>(policy), first, last, std::equal_to<>());
	}

	/// Copies [first, last) to `d_first`, but of every run of consecutive
	/// equivalent elements only the first one (parallel version of
	/// `unique_copy`): every thread counts the copied elements of its chunk,
	/// then every thread copies them to the prefix sum of the counts.
	/// NOTE: `p` must be an equivalence relation (as for `std::unique_copy`)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt1 random access iterator type of the source range
	/// \tparam RandIt2 random access iterator type of the destination range
	/// \tparam BinaryPred equivalence predicate type
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the destination range, must not
	/// 	overlap the source range
	/// \param p[in]: `p(a, b)` is true iff `a` and `b` are equivalent
	/// \return iterator past the last written element
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class BinaryPred,
	         const AlgorithmUniqueConfig &config = algorithmUniqueConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 unique_copy(ExecPolicy &&policy,
	                    RandIt1 first,
	                    RandIt1 last,
	                    RandIt2 d_first,
	                    BinaryPred p) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::unique_copy(first, last, d_first, p);
		}

		// element i is copied iff i == 0 or it is not equivalent to element i-1
		auto &task_pool = *policy.pool();
		std::vector<size_t> count(nthreads, 0), skip(nthreads, 0);
		std::vector<std::future<void>> futures;
		for (uint32_t k = 0; k < nthreads; ++k) {
			futures.emplace_back(task_pool.submit([&count, &skip, &p, first, n, k, nthreads]() {
				const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
				skip[k] = internal::unique_skip(first, n, k, nthreads, p);
				size_t c = (lo + skip[k] < hi) ? 1u : 0u;
				for (size_t i = lo + skip[k] + 1; i < hi; ++i) {
					c += p(first[i - 1], first[i]) ? 0u : 1u;
				}
				count[k] = c;
			}));
		}
		internal::get_futures(futures);
		futures.clear();

		size_t off = 0;
		for (uint32_t k = 0; k < nthreads; ++k) {
			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			futures.emplace_back(task_pool.submit([&p, b = first + lo + skip[k], e = first + hi, out = d_first + off]() {
				cryptanalysislib::unique_copy(b, e, out, p);
			}));
			off += count[k];
		}
		internal::get_futures(futures);
		return d_first + off;
	}

	/// same as above, compares the elements with `==`
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmUniqueConfig &config = algorithmUniqueConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 unique_copy(ExecPolicy &&policy,
	                    RandIt1 first,
	                    RandIt1 last,
	                    RandIt2 d_first) noexcept {
		return cryptanalysislib::unique_copy
			<ExecPolicy, RandIt1, RandIt2, std::equal_to<>, config>
			(std::forward<ExecPolicy>(policy), first, last, d_first, std::equal_to<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_UNIQUE_H
