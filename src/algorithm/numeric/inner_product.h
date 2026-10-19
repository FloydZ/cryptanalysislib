#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_INNER_PRODUCT_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_INNER_PRODUCT_H

#include <functional>
#include <future>
#include <iterator>
#include <utility>
#include <vector>

#include "algorithm/algorithm.h"

namespace cryptanalysislib {
	/// Computes `init + a[0]*b[0] + a[1]*b[1] + ...` in this order (like
	/// `std::inner_product`)
	/// NOTE: unlike `transform_reduce` the order of the operations is fixed
	/// 	and the result has the type of `init`
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param init[in]: initial value
	/// \return the inner product
	template<class InputIt1,
	         class InputIt2,
	         class T>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt1> &&
	             std::input_iterator<InputIt2>
#endif
	[[nodiscard]] constexpr T inner_product(InputIt1 first1,
	                                        InputIt1 last1,
	                                        InputIt2 first2,
	                                        T init) noexcept {
		for (; first1 != last1; ++first1, (void) ++first2) {
			init = std::move(init) + (*first1) * (*first2);
		}
		return init;
	}

	/// Computes `op1(...op1(op1(init, op2(a[0], b[0])), op2(a[1], b[1]))...)`
	/// (like `std::inner_product`)
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param init[in]: initial value
	/// \param op1[in]: "sum", combines the accumulator with a product
	/// \param op2[in]: "product", combines two elements
	/// \return the generalized inner product
	template<class InputIt1,
	         class InputIt2,
	         class T,
	         class BinaryOp1,
	         class BinaryOp2>
#if __cplusplus > 201709L
	    requires std::input_iterator<InputIt1> &&
	             std::input_iterator<InputIt2>
#endif
	[[nodiscard]] constexpr T inner_product(InputIt1 first1,
	                                        InputIt1 last1,
	                                        InputIt2 first2,
	                                        T init,
	                                        BinaryOp1 op1,
	                                        BinaryOp2 op2) noexcept {
		for (; first1 != last1; ++first1, (void) ++first2) {
			init = op1(std::move(init), op2(*first1, *first2));
		}
		return init;
	}

	/// Configuration for the parallel `inner_product`
	struct AlgorithmInnerProductConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 16u;
	};
	constexpr static AlgorithmInnerProductConfig algorithmInnerProductConfig;

	/// Computes `op1(...op1(op1(init, op2(a[0], b[0])), op2(a[1], b[1]))...)`
	/// (parallel version of `inner_product`, there is no `std::` one): every
	/// thread reduces its chunk starting with `T(op2(a[lo], b[lo]))`, the
	/// results of the chunks are combined in order into `init`.
	/// NOTE: unlike the sequential version `op1` must be associative (but not
	/// 	commutative, unlike `transform_reduce`) and accept two `T`
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param init[in]: initial value
	/// \param op1[in]: "sum", combines the accumulator with a product
	/// \param op2[in]: "product", combines two elements
	/// \return the generalized inner product
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class T,
	         class BinaryOp1,
	         class BinaryOp2,
	         const AlgorithmInnerProductConfig &config = algorithmInnerProductConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	[[nodiscard]] T inner_product(ExecPolicy &&policy,
	                              RandIt1 first1,
	                              RandIt1 last1,
	                              RandIt2 first2,
	                              T init,
	                              BinaryOp1 op1,
	                              BinaryOp2 op2) noexcept {
		const size_t n = static_cast<size_t>(std::distance(first1, last1));
		const uint32_t nthreads = should_par(policy, config, n);
		if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
			return cryptanalysislib::inner_product(first1, last1, first2, std::move(init), op1, op2);
		}

		auto &task_pool = *policy.pool();
		std::vector<std::future<T>> futures;
		for (uint32_t k = 0; k < nthreads; ++k) {
			const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
			futures.emplace_back(task_pool.submit([&op1, &op2, first1, first2, lo, hi]() -> T {
				return cryptanalysislib::inner_product(first1 + lo + 1, first1 + hi, first2 + lo + 1,
				                                       static_cast<T>(op2(first1[lo], first2[lo])), op1, op2);
			}));
		}

		for (auto &f : futures) {
			init = op1(std::move(init), f.get());
		}
		return init;
	}

	/// same as above, with `+` and `*`
	template<class ExecPolicy,
	         class RandIt1,
	         class RandIt2,
	         class T,
	         const AlgorithmInnerProductConfig &config = algorithmInnerProductConfig>
#if __cplusplus > 201709L
	    requires ExecutionPolicy<ExecPolicy> &&
	             std::random_access_iterator<RandIt1> &&
	             std::random_access_iterator<RandIt2>
#endif
	[[nodiscard]] T inner_product(ExecPolicy &&policy,
	                              RandIt1 first1,
	                              RandIt1 last1,
	                              RandIt2 first2,
	                              T init) noexcept {
		return cryptanalysislib::inner_product
			<ExecPolicy, RandIt1, RandIt2, T, std::plus<>, std::multiplies<>, config>
			(std::forward<ExecPolicy>(policy), first1, last1, first2, std::move(init), std::plus<>(), std::multiplies<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_INNER_PRODUCT_H
