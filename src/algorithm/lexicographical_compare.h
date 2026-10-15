#ifndef CRYPTANALYSISLIB_ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
#define CRYPTANALYSISLIB_ALGORITHM_LEXICOGRAPHICAL_COMPARE_H

#include <compare>
#include <cstdint>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "algorithm/mismatch.h"

namespace cryptanalysislib {
	namespace internal {
		/// true if the SIMD `mismatch_simd_uXX` can be used to find the first
		/// position in which the two ranges differ
		template<class InputIt1, class InputIt2, class Compare, class Default>
		constexpr static bool lexicographical_compare_simd =
		        std::contiguous_iterator<InputIt1> &&
		        std::contiguous_iterator<InputIt2> &&
		        std::is_same_v<std::iter_value_t<InputIt1>, std::iter_value_t<InputIt2>> &&
		        (std::is_same_v<std::iter_value_t<InputIt1>, uint8_t> ||
		         std::is_same_v<std::iter_value_t<InputIt1>, uint16_t> ||
		         std::is_same_v<std::iter_value_t<InputIt1>, uint32_t> ||
		         std::is_same_v<std::iter_value_t<InputIt1>, uint64_t>) &&
		        std::is_same_v<Compare, Default>;
	} // end namespace internal

	/// Checks if [first1, last1) is lexicographically less than [first2,
	/// last2) w.r.t. `comp` (like `std::lexicographical_compare`): the first
	/// non equivalent pair decides, otherwise the shorter range is less.
	/// NOTE: contiguous `uint8_t`...`uint64_t` ranges and `std::less<>` use
	/// 	the SIMD `mismatch` kernel.
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param last2[in]: end of the second range
	/// \param comp[in]: less-than comparison
	/// \return true if the first range is less than the second one
	template<class InputIt1,
	         class InputIt2,
	         class Compare>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt1> &&
	             std::input_iterator<InputIt2>
#endif
	[[nodiscard]] constexpr bool lexicographical_compare(InputIt1 first1,
	                                                     InputIt1 last1,
	                                                     InputIt2 first2,
	                                                     InputIt2 last2,
	                                                     Compare comp) noexcept {
		if constexpr (internal::lexicographical_compare_simd<InputIt1, InputIt2, Compare, std::less<>>) {
			if !consteval {
				using T = std::iter_value_t<InputIt1>;
				const size_t n1 = static_cast<size_t>(last1 - first1);
				const size_t n2 = static_cast<size_t>(last2 - first2);
				const size_t n = n1 < n2 ? n1 : n2;
				const size_t k = internal::mismatch_simd_uXX<T>(std::to_address(first1), std::to_address(first2), n);
				if (k < n) {
					return first1[k] < first2[k];
				}
				return n1 < n2;
			}
		}

		for (; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2) {
			if (comp(*first1, *first2)) {
				return true;
			}
			if (comp(*first2, *first1)) {
				return false;
			}
		}
		return (first1 == last1) && (first2 != last2);
	}

	/// same as above, with `<`
	template<class InputIt1,
	         class InputIt2>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt1> &&
	             std::input_iterator<InputIt2>
#endif
	[[nodiscard]] constexpr bool lexicographical_compare(InputIt1 first1,
	                                                     InputIt1 last1,
	                                                     InputIt2 first2,
	                                                     InputIt2 last2) noexcept {
		return cryptanalysislib::lexicographical_compare(first1, last1, first2, last2, std::less<>());
	}

	/// Compares [first1, last1) and [first2, last2) lexicographically with the
	/// three way comparison `comp` (like `std::lexicographical_compare_three_way`)
	/// NOTE: contiguous `uint8_t`...`uint64_t` ranges and
	/// 	`std::compare_three_way` use the SIMD `mismatch` kernel.
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param last2[in]: end of the second range
	/// \param comp[in]: three way comparison, returns a comparison category
	/// \return the result of `comp` on the first non equal pair, otherwise
	/// 	the comparison of the lengths
	template<class InputIt1,
	         class InputIt2,
	         class Cmp>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt1> &&
	             std::input_iterator<InputIt2>
#endif
	[[nodiscard]] constexpr auto lexicographical_compare_three_way(InputIt1 first1,
	                                                               InputIt1 last1,
	                                                               InputIt2 first2,
	                                                               InputIt2 last2,
	                                                               Cmp comp) noexcept
	        -> decltype(comp(*first1, *first2)) {
		using R = decltype(comp(*first1, *first2));
		static_assert(std::is_same_v<R, std::strong_ordering> ||
		              std::is_same_v<R, std::weak_ordering> ||
		              std::is_same_v<R, std::partial_ordering>);

		if constexpr (internal::lexicographical_compare_simd<InputIt1, InputIt2, Cmp, std::compare_three_way>) {
			if !consteval {
				using T = std::iter_value_t<InputIt1>;
				const size_t n1 = static_cast<size_t>(last1 - first1);
				const size_t n2 = static_cast<size_t>(last2 - first2);
				const size_t n = n1 < n2 ? n1 : n2;
				const size_t k = internal::mismatch_simd_uXX<T>(std::to_address(first1), std::to_address(first2), n);
				if (k < n) {
					return first1[k] <=> first2[k];
				}
				return n1 <=> n2;
			}
		}

		bool exhausted1 = (first1 == last1);
		bool exhausted2 = (first2 == last2);
		for (; !exhausted1 && !exhausted2; exhausted1 = (++first1 == last1), exhausted2 = (++first2 == last2)) {
			if (const auto c = comp(*first1, *first2); c != 0) {
				return c;
			}
		}

		return !exhausted1 ? std::strong_ordering::greater
		                   : (!exhausted2 ? std::strong_ordering::less : std::strong_ordering::equal);
	}

	/// same as above, with `<=>`
	template<class InputIt1,
	         class InputIt2>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt1> &&
	             std::input_iterator<InputIt2>
#endif
	[[nodiscard]] constexpr auto lexicographical_compare_three_way(InputIt1 first1,
	                                                               InputIt1 last1,
	                                                               InputIt2 first2,
	                                                               InputIt2 last2) noexcept {
		return cryptanalysislib::lexicographical_compare_three_way(first1, last1, first2, last2,
		                                                           std::compare_three_way());
	}

	/// Configuration for the parallel `lexicographical_compare`
	/// NOTE: the comparison usually stops early, so only long common
	/// 	prefixes pay off.
	struct AlgorithmLexicographicalCompareConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 18u;
	};
	constexpr static AlgorithmLexicographicalCompareConfig algorithmLexicographicalCompareConfig;

	/// Checks if [first1, last1) is lexicographically less than [first2,
	/// last2) w.r.t. `comp` (parallel version of `lexicographical_compare`):
	/// every thread searches the first non equivalent pair in its chunk of
	/// the common length, the first one found decides.
	/// NOTE: contiguous `uint8_t`...`uint64_t` ranges and `std::less<>` use
	/// 	the SIMD `mismatch` kernel in every chunk.
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param last2[in]: end of the second range
	/// \param comp[in]: less-than comparison
	/// \return true if the first range is less than the second one
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class Compare,
	         const AlgorithmLexicographicalCompareConfig &config = algorithmLexicographicalCompareConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	[[nodiscard]] bool lexicographical_compare(ExecPolicy &&policy,
	                                           RandIt1 first1,
	                                           RandIt1 last1,
	                                           RandIt2 first2,
	                                           RandIt2 last2,
	                                           Compare comp) noexcept {
		const size_t n1 = static_cast<size_t>(last1 - first1);
		const size_t n2 = static_cast<size_t>(last2 - first2);
		const size_t n = n1 < n2 ? n1 : n2;
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::lexicographical_compare(first1, last1, first2, last2, comp);
		}

		// index of the first non equivalent pair of every chunk, or n
		auto &task_pool = *policy.pool();
		std::vector<std::future<size_t>> futures;
		for (uint32_t k = 0; k < nthreads; ++k) {
			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			futures.emplace_back(task_pool.submit([&comp, first1, first2, lo, hi, n]() -> size_t {
				if constexpr (internal::lexicographical_compare_simd<RandIt1, RandIt2, Compare, std::less<>>) {
					using T = std::iter_value_t<RandIt1>;
					(void) comp;
					const size_t i = internal::mismatch_simd_uXX<T>(std::to_address(first1) + lo,
					                                                std::to_address(first2) + lo, hi - lo);
					return (i < hi - lo) ? lo + i : n;
				} else {
					for (size_t i = lo; i < hi; ++i) {
						if (comp(first1[i], first2[i]) || comp(first2[i], first1[i])) {
							return i;
						}
					}
					return n;
				}
			}));
		}

		size_t k = n;
		for (auto &f : futures) {
			const size_t i = f.get();
			k = i < k ? i : k;
		}

		if (k < n) {
			return comp(first1[k], first2[k]);
		}
		return n1 < n2;
	}

	/// same as above, with `<`
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         const AlgorithmLexicographicalCompareConfig &config = algorithmLexicographicalCompareConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	[[nodiscard]] bool lexicographical_compare(ExecPolicy &&policy,
	                                           RandIt1 first1,
	                                           RandIt1 last1,
	                                           RandIt2 first2,
	                                           RandIt2 last2) noexcept {
		return cryptanalysislib::lexicographical_compare
			<ExecPolicy, RandIt1, RandIt2, std::less<>, config>
			(std::forward<ExecPolicy>(policy), first1, last1, first2, last2, std::less<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
