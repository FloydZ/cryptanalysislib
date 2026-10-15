#ifndef CRYPTANALYSISLIB_ALGORITHM_UNINITIALIZED_H
#define CRYPTANALYSISLIB_ALGORITHM_UNINITIALIZED_H

#include <future>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "algorithm/algorithm.h"

/// Algorithms on uninitialized memory (like `std::uninitialized_copy`,
/// `std::destroy`, ... from `<memory>`).
/// NOTE: all of them are `noexcept`: a throwing constructor terminates the
/// 	program, so unlike the `std::` versions nothing is rolled back.
namespace cryptanalysislib {
	/// Creates a `T` from `args` at the address `p` (like `std::construct_at`)
	/// NOTE: forwards to `std::construct_at`, the only way to begin the
	/// 	lifetime of an object in a constant expression (C++20)
	/// \param p[in]: uninitialized storage for a `T`
	/// \param args[in]: constructor arguments
	/// \return pointer to the new object
	template<class T,
	         class... Args>
	constexpr T *construct_at(T *p,
	                          Args &&...args) noexcept {
		return std::construct_at(p, std::forward<Args>(args)...);
	}

	/// Calls the destructor of `*p`, of every element for arrays (like
	/// `std::destroy_at`)
	/// \param p[in]: object to destroy
	template<class T>
	constexpr void destroy_at(T *p) noexcept {
		if constexpr (std::is_array_v<T>) {
			for (auto &e : *p) {
				cryptanalysislib::destroy_at(std::addressof(e));
			}
		} else {
			p->~T();
		}
	}

	/// Destroys the objects in [first, last) (like `std::destroy`)
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	template<class ForwardIt>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr void destroy(ForwardIt first,
	                       ForwardIt last) noexcept {
		if constexpr (!std::is_trivially_destructible_v<std::iter_value_t<ForwardIt>>) {
			for (; first != last; ++first) {
				cryptanalysislib::destroy_at(std::addressof(*first));
			}
		}
	}

	/// Destroys the `n` objects starting at `first` (like `std::destroy_n`)
	/// \param first[in]: beginning of the range
	/// \param n[in]: number of objects
	/// \return iterator past the last destroyed object
	template<class ForwardIt,
	         class Size>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt destroy_n(ForwardIt first,
	                              Size n) noexcept {
		for (; n > 0; ++first, (void) --n) {
			cryptanalysislib::destroy_at(std::addressof(*first));
		}
		return first;
	}

	/// Copy constructs the elements of [first, last) into the uninitialized
	/// range starting at `d_first` (like `std::uninitialized_copy`)
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the uninitialized destination range
	/// \return iterator past the last constructed element
	template<class InputIt,
	         class ForwardIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt> &&
	             std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt uninitialized_copy(InputIt first,
	                                       InputIt last,
	                                       ForwardIt d_first) noexcept {
		for (; first != last; ++first, (void) ++d_first) {
			cryptanalysislib::construct_at(std::addressof(*d_first), *first);
		}
		return d_first;
	}

	/// Copy constructs `n` elements starting at `first` into the
	/// uninitialized range starting at `d_first` (like `std::uninitialized_copy_n`)
	/// \param first[in]: beginning of the source range
	/// \param n[in]: number of elements
	/// \param d_first[out]: beginning of the uninitialized destination range
	/// \return iterator past the last constructed element
	template<class InputIt,
	         class Size,
	         class ForwardIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt> &&
	             std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt uninitialized_copy_n(InputIt first,
	                                         Size n,
	                                         ForwardIt d_first) noexcept {
		for (; n > 0; ++first, (void) ++d_first, --n) {
			cryptanalysislib::construct_at(std::addressof(*d_first), *first);
		}
		return d_first;
	}

	/// Move constructs the elements of [first, last) into the uninitialized
	/// range starting at `d_first` (like `std::uninitialized_move`)
	/// \param first[in]: beginning of the source range
	/// \param last[in]: end of the source range
	/// \param d_first[out]: beginning of the uninitialized destination range
	/// \return iterator past the last constructed element
	template<class InputIt,
	         class ForwardIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt> &&
	             std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt uninitialized_move(InputIt first,
	                                       InputIt last,
	                                       ForwardIt d_first) noexcept {
		for (; first != last; ++first, (void) ++d_first) {
			cryptanalysislib::construct_at(std::addressof(*d_first), std::move(*first));
		}
		return d_first;
	}

	/// Move constructs `n` elements starting at `first` into the
	/// uninitialized range starting at `d_first` (like `std::uninitialized_move_n`)
	/// \param first[in]: beginning of the source range
	/// \param n[in]: number of elements
	/// \param d_first[out]: beginning of the uninitialized destination range
	/// \return {past the last moved from element, past the last constructed element}
	template<class InputIt,
	         class Size,
	         class ForwardIt>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt> &&
	             std::forward_iterator<ForwardIt>
#endif
	constexpr std::pair<InputIt, ForwardIt> uninitialized_move_n(InputIt first,
	                                                             Size n,
	                                                             ForwardIt d_first) noexcept {
		for (; n > 0; ++first, (void) ++d_first, --n) {
			cryptanalysislib::construct_at(std::addressof(*d_first), std::move(*first));
		}
		return {first, d_first};
	}

	/// Copy constructs `value` into every element of the uninitialized range
	/// [first, last) (like `std::uninitialized_fill`)
	/// \param first[in]: beginning of the uninitialized range
	/// \param last[in]: end of the uninitialized range
	/// \param value[in]: value to copy
	template<class ForwardIt,
	         class T>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr void uninitialized_fill(ForwardIt first,
	                                  ForwardIt last,
	                                  const T &value) noexcept {
		for (; first != last; ++first) {
			cryptanalysislib::construct_at(std::addressof(*first), value);
		}
	}

	/// Copy constructs `value` into the `n` uninitialized elements starting
	/// at `first` (like `std::uninitialized_fill_n`)
	/// \param first[in]: beginning of the uninitialized range
	/// \param n[in]: number of elements
	/// \param value[in]: value to copy
	/// \return iterator past the last constructed element
	template<class ForwardIt,
	         class Size,
	         class T>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt uninitialized_fill_n(ForwardIt first,
	                                         Size n,
	                                         const T &value) noexcept {
		for (; n > 0; ++first, (void) --n) {
			cryptanalysislib::construct_at(std::addressof(*first), value);
		}
		return first;
	}

	/// Default initializes (`new T`, e.g. no zeroing of integers) the objects
	/// in the uninitialized range [first, last) (like
	/// `std::uninitialized_default_construct`)
	/// \param first[in]: beginning of the uninitialized range
	/// \param last[in]: end of the uninitialized range
	template<class ForwardIt>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	void uninitialized_default_construct(ForwardIt first,
	                                     ForwardIt last) noexcept {
		using T = std::iter_value_t<ForwardIt>;
		if constexpr (!std::is_trivially_default_constructible_v<T>) {
			for (; first != last; ++first) {
				::new (static_cast<void *>(std::addressof(*first))) T;
			}
		}
	}

	/// Default initializes the `n` objects starting at `first` (like
	/// `std::uninitialized_default_construct_n`)
	/// \param first[in]: beginning of the uninitialized range
	/// \param n[in]: number of objects
	/// \return iterator past the last constructed object
	template<class ForwardIt,
	         class Size>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	ForwardIt uninitialized_default_construct_n(ForwardIt first,
	                                            Size n) noexcept {
		using T = std::iter_value_t<ForwardIt>;
		for (; n > 0; ++first, (void) --n) {
			::new (static_cast<void *>(std::addressof(*first))) T;
		}
		return first;
	}

	/// Value initializes (`T()`, e.g. zeroes integers) the objects in the
	/// uninitialized range [first, last) (like `std::uninitialized_value_construct`)
	/// \param first[in]: beginning of the uninitialized range
	/// \param last[in]: end of the uninitialized range
	template<class ForwardIt>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr void uninitialized_value_construct(ForwardIt first,
	                                             ForwardIt last) noexcept {
		for (; first != last; ++first) {
			cryptanalysislib::construct_at(std::addressof(*first));
		}
	}

	/// Value initializes the `n` objects starting at `first` (like
	/// `std::uninitialized_value_construct_n`)
	/// \param first[in]: beginning of the uninitialized range
	/// \param n[in]: number of objects
	/// \return iterator past the last constructed object
	template<class ForwardIt,
	         class Size>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr ForwardIt uninitialized_value_construct_n(ForwardIt first,
	                                                    Size n) noexcept {
		for (; n > 0; ++first, (void) --n) {
			cryptanalysislib::construct_at(std::addressof(*first));
		}
		return first;
	}

	/// Configuration for the parallel algorithms on uninitialized memory
	struct AlgorithmUninitializedConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 15u;
	};
	constexpr static AlgorithmUninitializedConfig algorithmUninitializedConfig;

	namespace internal {
		/// runs `chunk(lo, hi)` for each of the `nthreads` chunks of [0, n) in
		/// parallel and waits for all of them
		template<class ExecPolicy, class Chunk>
		void uninitialized_parallel(ExecPolicy &&policy,
		                            const size_t n,
		                            const uint32_t nthreads,
		                            Chunk chunk) noexcept {
			auto &task_pool = *policy.pool();
			std::vector<std::future<void>> futures;
			for (uint32_t k = 0; k < nthreads; ++k) {
				const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
				futures.emplace_back(task_pool.submit([&chunk, lo, hi]() { chunk(lo, hi); }));
			}
			internal::get_futures(futures);
		}
	} // end namespace internal

	/// parallel version of `uninitialized_copy`: every thread constructs one chunk
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 uninitialized_copy(ExecPolicy &&policy,
	                           RandIt1 first,
	                           RandIt1 last,
	                           RandIt2 d_first) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::uninitialized_copy(first, last, d_first);
		}

		internal::uninitialized_parallel(policy, n, nthreads, [first, d_first](const size_t lo, const size_t hi) {
			cryptanalysislib::uninitialized_copy(first + lo, first + hi, d_first + lo);
		});
		return d_first + n;
	}

	/// parallel version of `uninitialized_copy_n`
	template<class ExecPolicy,
	         class RandIt1,
	         class Size,
	         class RandIt2,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 uninitialized_copy_n(ExecPolicy &&policy,
	                             RandIt1 first,
	                             Size count,
	                             RandIt2 d_first) noexcept {
		if (count <= 0) {
			return d_first;
		}
		return cryptanalysislib::uninitialized_copy
			<ExecPolicy, RandIt1, RandIt2, config>
			(std::forward<ExecPolicy>(policy), first, first + count, d_first);
	}

	/// parallel version of `uninitialized_move`: every thread constructs one chunk
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	RandIt2 uninitialized_move(ExecPolicy &&policy,
	                           RandIt1 first,
	                           RandIt1 last,
	                           RandIt2 d_first) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::uninitialized_move(first, last, d_first);
		}

		internal::uninitialized_parallel(policy, n, nthreads, [first, d_first](const size_t lo, const size_t hi) {
			cryptanalysislib::uninitialized_move(first + lo, first + hi, d_first + lo);
		});
		return d_first + n;
	}

	/// parallel version of `uninitialized_move_n`
	/// \return {past the last moved from element, past the last constructed element}
	template<class ExecPolicy,
	         class RandIt1,
	         class Size,
	         class RandIt2,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	std::pair<RandIt1, RandIt2> uninitialized_move_n(ExecPolicy &&policy,
	                                                 RandIt1 first,
	                                                 Size count,
	                                                 RandIt2 d_first) noexcept {
		if (count <= 0) {
			return {first, d_first};
		}
		const RandIt2 d_last = cryptanalysislib::uninitialized_move
			<ExecPolicy, RandIt1, RandIt2, config>
			(std::forward<ExecPolicy>(policy), first, first + count, d_first);
		return {first + count, d_last};
	}

	/// parallel version of `uninitialized_fill`: every thread constructs one chunk
	template<class ExecPolicy,
	         class RandIt,
	         class T,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	void uninitialized_fill(ExecPolicy &&policy,
	                        RandIt first,
	                        RandIt last,
	                        const T &value) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			cryptanalysislib::uninitialized_fill(first, last, value);
			return;
		}

		internal::uninitialized_parallel(policy, n, nthreads, [first, &value](const size_t lo, const size_t hi) {
			cryptanalysislib::uninitialized_fill(first + lo, first + hi, value);
		});
	}

	/// parallel version of `uninitialized_fill_n`
	/// \return iterator past the last constructed element
	template<class ExecPolicy,
	         class RandIt,
	         class Size,
	         class T,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	RandIt uninitialized_fill_n(ExecPolicy &&policy,
	                            RandIt first,
	                            Size count,
	                            const T &value) noexcept {
		if (count <= 0) {
			return first;
		}
		cryptanalysislib::uninitialized_fill
			<ExecPolicy, RandIt, T, config>
			(std::forward<ExecPolicy>(policy), first, first + count, value);
		return first + count;
	}

	/// parallel version of `uninitialized_default_construct`: every thread
	/// constructs one chunk
	template<class ExecPolicy,
	         class RandIt,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	void uninitialized_default_construct(ExecPolicy &&policy,
	                                     RandIt first,
	                                     RandIt last) noexcept {
		if constexpr (std::is_trivially_default_constructible_v<std::iter_value_t<RandIt>>) {
			return;
		}

		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			cryptanalysislib::uninitialized_default_construct(first, last);
			return;
		}

		internal::uninitialized_parallel(policy, n, nthreads, [first](const size_t lo, const size_t hi) {
			cryptanalysislib::uninitialized_default_construct(first + lo, first + hi);
		});
	}

	/// parallel version of `uninitialized_default_construct_n`
	/// \return iterator past the last constructed object
	template<class ExecPolicy,
	         class RandIt,
	         class Size,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	RandIt uninitialized_default_construct_n(ExecPolicy &&policy,
	                                         RandIt first,
	                                         Size count) noexcept {
		if (count <= 0) {
			return first;
		}
		cryptanalysislib::uninitialized_default_construct
			<ExecPolicy, RandIt, config>
			(std::forward<ExecPolicy>(policy), first, first + count);
		return first + count;
	}

	/// parallel version of `uninitialized_value_construct`: every thread
	/// constructs one chunk
	template<class ExecPolicy,
	         class RandIt,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	void uninitialized_value_construct(ExecPolicy &&policy,
	                                   RandIt first,
	                                   RandIt last) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			cryptanalysislib::uninitialized_value_construct(first, last);
			return;
		}

		internal::uninitialized_parallel(policy, n, nthreads, [first](const size_t lo, const size_t hi) {
			cryptanalysislib::uninitialized_value_construct(first + lo, first + hi);
		});
	}

	/// parallel version of `uninitialized_value_construct_n`
	/// \return iterator past the last constructed object
	template<class ExecPolicy,
	         class RandIt,
	         class Size,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	RandIt uninitialized_value_construct_n(ExecPolicy &&policy,
	                                       RandIt first,
	                                       Size count) noexcept {
		if (count <= 0) {
			return first;
		}
		cryptanalysislib::uninitialized_value_construct
			<ExecPolicy, RandIt, config>
			(std::forward<ExecPolicy>(policy), first, first + count);
		return first + count;
	}

	/// parallel version of `destroy`: every thread destroys one chunk
	template<class ExecPolicy,
	         class RandIt,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	void destroy(ExecPolicy &&policy,
	             RandIt first,
	             RandIt last) noexcept {
		if constexpr (std::is_trivially_destructible_v<std::iter_value_t<RandIt>>) {
			return;
		}

		const size_t n = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			cryptanalysislib::destroy(first, last);
			return;
		}

		internal::uninitialized_parallel(policy, n, nthreads, [first](const size_t lo, const size_t hi) {
			cryptanalysislib::destroy(first + lo, first + hi);
		});
	}

	/// parallel version of `destroy_n`
	/// \return iterator past the last destroyed object
	template<class ExecPolicy,
	         class RandIt,
	         class Size,
	         const AlgorithmUninitializedConfig &config = algorithmUninitializedConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt>
#endif
	RandIt destroy_n(ExecPolicy &&policy,
	                 RandIt first,
	                 Size count) noexcept {
		if (count <= 0) {
			return first;
		}
		cryptanalysislib::destroy
			<ExecPolicy, RandIt, config>
			(std::forward<ExecPolicy>(policy), first, first + count);
		return first + count;
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_UNINITIALIZED_H
