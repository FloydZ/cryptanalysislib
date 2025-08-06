#ifndef CRYPTANALYSISLIB_ALGORITHM_TRANSFORM_H
#define CRYPTANALYSISLIB_ALGORITHM_TRANSFORM_H

#include <numeric>

#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {

	struct AlgorithmTransformConfig : public AlgorithmConfig {
		const size_t min_size_per_thread = 1u << 10u;
	};
	constexpr static AlgorithmTransformConfig algorithmTransformConfig;

	/// Applies a unary operation to each element in the input range and stores the result in the output range
	///
	/// \tparam InputIt type of the input iterator
	/// \tparam OutputIt type of the output iterator
	/// \tparam UnaryOperation type of the unary operation
	/// \tparam config configuration for the algorithm
	/// \param first1[in]: iterator to the first element in the input range
	/// \param last1[in]: iterator to one past the last element in the input range
	/// \param d_first[out]: iterator to the first element in the output range
	/// \param unary_op[in]: unary operation to apply to each element
	/// \return iterator to the element past the last element written
	template<class InputIt,
			 class OutputIt,
			 class UnaryOperation,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
				 std::forward_iterator<OutputIt> &&
    			 std::regular_invocable<UnaryOperation,
										typename InputIt::value_type&>
#endif
	constexpr OutputIt transform(InputIt first1,
								 InputIt last1,
								 OutputIt d_first,
								 UnaryOperation unary_op) noexcept {
		for (; first1 != last1; ++d_first, ++first1) {
			*d_first = unary_op(*first1);
		}

		return d_first;
	}

	/// Applies a binary operation to pairs of elements from two ranges and stores the result in the output range
	///
	/// \tparam InputIt1 type of the first input iterator
	/// \tparam InputIt2 type of the second input iterator
	/// \tparam OutputIt type of the output iterator
	/// \tparam UnaryOperation type of the binary operation (despite the name)
	/// \tparam config configuration for the algorithm
	/// \param first1[in]: iterator to the first element in the first input range
	/// \param last1[in]: iterator to one past the last element in the first input range
	/// \param first2[in]: iterator to the first element in the second input range
	/// \param d_first[out]: iterator to the first element in the output range
	/// \param binary_op[in]: binary operation to apply to pairs of elements
	/// \return iterator to the element past the last element written
	template<class InputIt1,
			 class InputIt2,
			 class OutputIt,
			 class UnaryOperation,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt1> &&
				 std::random_access_iterator<InputIt2> &&
    			 std::regular_invocable<UnaryOperation,
										typename InputIt1::value_type&,
										typename InputIt1::value_type&>
#endif
	constexpr OutputIt transform(InputIt1 first1,
								 InputIt1 last1,
								 InputIt2 first2,
								 OutputIt d_first,
								 UnaryOperation binary_op) noexcept {
		for (; first1 != last1; ++d_first, ++first1, ++first2) {
			*d_first = binary_op(*first1, *first2);
		}

		return d_first;
	}

	/// Transforms and reduces two ranges using the specified operations
	///
	/// \tparam ForwardIt1 type of the first input iterator
	/// \tparam ForwardIt2 type of the second input iterator
	/// \tparam BinaryOp1 type of the reduction operation
	/// \tparam BinaryOp2 type of the transformation operation
	/// \tparam config configuration for the algorithm
	/// \param first1[in]: iterator to the first element in the first input range
	/// \param last1[in]: iterator to one past the last element in the first input range
	/// \param first2[in]: iterator to the first element in the second input range
	/// \param init[in]: initial value for the reduction
	/// \param reduce[in]: binary reduction operation
	/// \param transform[in]: binary transformation operation
	/// \return result of the transform-reduce operation
	template<class ForwardIt1,
			 class ForwardIt2,
			 class BinaryOp1,
			 class BinaryOp2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<ForwardIt1> &&
				 std::forward_iterator<ForwardIt2> &&
    			 std::regular_invocable<BinaryOp1,
										typename ForwardIt1::value_type&,
										typename ForwardIt1::value_type&> &&
    			 std::regular_invocable<BinaryOp2,
										typename ForwardIt1::value_type&,
										typename ForwardIt1::value_type&>
#endif
	ForwardIt1::value_type transform_reduce(ForwardIt1 first1,
											ForwardIt1 last1,
											ForwardIt2 first2,
											const typename ForwardIt1::value_type init,
										    BinaryOp1 reduce,
										    BinaryOp2 transform) noexcept {
		using T = ForwardIt1::value_type;
		T ret = init;
		for (; first1 != last1; ++first1, ++first2) {
			ret = reduce(transform(*first1, *first2), ret);
		}

		return ret;
	}


	/// Transforms and reduces two ranges using multiplication and addition operations
	///
	/// \tparam InputIt1 type of the first input iterator
	/// \tparam InputIt2 type of the second input iterator
	/// \tparam config configuration for the algorithm
	/// \param first1[in]: iterator to the first element in the first input range
	/// \param last1[in]: iterator to one past the last element in the first input range
	/// \param first2[in]: iterator to the first element in the second input range
	/// \param init[in]: initial value for the reduction
	/// \return result of the transform-reduce operation (inner product)
	template<class InputIt1,
			 class InputIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt1> &&
				 std::forward_iterator<InputIt2>
#endif
	InputIt1::value_type transform_reduce(InputIt1 first1,
										  InputIt1 last1,
										  InputIt2 first2,
										  typename InputIt1::value_type init) noexcept {
		using T = InputIt1::value_type;
		return transform_reduce
				<InputIt1, InputIt2, decltype(std::plus<T>()), decltype(std::multiplies<T>()), config>
				(first1, last1, first2, init, std::plus<T>(), std::multiplies<T>());
	}


	/// Transforms each element in a range and then reduces the results
	///
	/// \tparam InputIt type of the input iterator
	/// \tparam BinaryOp type of the binary reduction operation
	/// \tparam UnaryOp type of the unary transformation operation
	/// \tparam config configuration for the algorithm
	/// \param first[in]: iterator to the first element in the input range
	/// \param last[in]: iterator to one past the last element in the input range
	/// \param init[in]: initial value for the reduction
	/// \param reduce[in]: binary reduction operation
	/// \param transform[in]: unary transformation operation
	/// \return result of the transform-reduce operation
	template<class InputIt,
             class BinaryOp,
			 class UnaryOp,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
    			 std::regular_invocable<BinaryOp,
										typename InputIt::value_type&> &&
    			 std::regular_invocable<UnaryOp,
										typename InputIt::value_type&,
										typename InputIt::value_type&>
#endif
	InputIt::value_type transform_reduce(InputIt first,
										 InputIt last,
										 const typename InputIt::value_type init,
										 BinaryOp reduce,
										 UnaryOp transform) noexcept {
		using T = InputIt::value;
		T ret = init;
		for (; first != last; ++first) {
			ret = reduce(transform(first), ret);
		}

		return ret;
	}

	/// Parallel transform-reduce operation that uses an execution policy
	///
	/// \tparam ExecPolicy type of the execution policy
	/// \tparam RandIt1 type of the random access iterator
	/// \tparam BinaryReductionOp type of the binary reduction operation
	/// \tparam UnaryTransformOp type of the unary transformation operation
	/// \tparam config configuration for the algorithm
	/// \param policy[in]: execution policy
	/// \param first1[in]: iterator to the first element in the input range
	/// \param last1[in]: iterator to one past the last element in the input range
	/// \param init[in]: initial value for the reduction
	/// \param reduce_op[in]: binary reduction operation
	/// \param transform_op[in]: unary transformation operation
	/// \return result of the transform-reduce operation
	template <class ExecPolicy,
			  class RandIt1,
			  class BinaryReductionOp,
			  class UnaryTransformOp,
			  const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
    			 std::regular_invocable<BinaryReductionOp,
										typename RandIt1::value_type&,
										typename RandIt1::value_type&> &&
    			 std::regular_invocable<UnaryTransformOp,
										typename RandIt1::value_type&,
										typename RandIt1::value_type&>
#endif
	typename RandIt1::value_value
	transform_reduce(ExecPolicy&& policy,
					 RandIt1 first1,
					 RandIt1 last1,
					 const typename RandIt1::value_value init,
					 BinaryReductionOp reduce_op,
					 UnaryTransformOp transform_op) noexcept {
		using T = RandIt1::value_type;
		const auto size = static_cast<size_t>(std::distance(first1, last1));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::transform_reduce
				<RandIt1, RandIt1, BinaryReductionOp, UnaryTransformOp, config>
				(first1, last1, init, reduce_op, transform_op);
		}

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy), first1, last1,
			std::transform_reduce<RandIt1, T,
			BinaryReductionOp, UnaryTransformOp>,
			(T*)nullptr,
			1,
			nthreads,
			init, reduce_op, transform_op);

		return std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), init, reduce_op);
	}

	/// Parallel transform-reduce operation for two ranges that uses an execution policy
	///
	/// \tparam ExecPolicy type of the execution policy
	/// \tparam RandIt1 type of the first random access iterator
	/// \tparam RandIt2 type of the second random access iterator
	/// \tparam BinaryReductionOp type of the binary reduction operation
	/// \tparam BinaryTransformOp type of the binary transformation operation
	/// \tparam config configuration for the algorithm
	template <class ExecPolicy,
			  class RandIt1,
			  class RandIt2,
			  class BinaryReductionOp,
			  class BinaryTransformOp,
			  const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
				 std::random_access_iterator<RandIt2> &&
    			 std::regular_invocable<BinaryReductionOp,
										typename RandIt1::value_type&,
										typename RandIt1::value_type&> &&
    			 std::regular_invocable<BinaryTransformOp,
										typename RandIt1::value_type&,
										typename RandIt1::value_type&>
#endif
	/// Performs a parallel transform-reduce operation on two ranges
	///
	/// \param policy[in]: execution policy
	/// \param first1[in]: iterator to the first element in the first input range
	/// \param last1[in]: iterator to one past the last element in the first input range
	/// \param first2[in]: iterator to the first element in the second input range
	/// \param init[in]: initial value for the reduction
	/// \param reduce_op[in]: binary reduction operation
	/// \param transform_op[in]: binary transformation operation
	/// \return result of the transform-reduce operation
	RandIt1::value_type
	transform_reduce(ExecPolicy&& policy,
					 RandIt1 first1,
					 RandIt1 last1,
					 RandIt2 first2,
					 const typename RandIt1::value_type init,
					 BinaryReductionOp reduce_op,
					 BinaryTransformOp transform_op) noexcept {
		using T = RandIt1::value_type;
		const auto size = static_cast<size_t>(std::distance(first1, last1));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::transform_reduce(first1, last1, first2, init, reduce_op, transform_op);
		}

		auto futures = internal::parallel_chunk_for_2(
			std::forward<ExecPolicy>(policy), first1, last1, first2,
				 cryptanalysislib::transform_reduce<RandIt1, RandIt2, BinaryReductionOp, BinaryTransformOp, config>,
				(T*)nullptr,
				nthreads,
				init, reduce_op, transform_op);

		return std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), init, reduce_op);
	}

	/// Parallel transform-reduce operation for two ranges using default addition and multiplication
	///
	/// \tparam ExecPolicy type of the execution policy
	/// \tparam RandIt1 type of the first random access iterator
	/// \tparam RandIt2 type of the second random access iterator
	/// \tparam config configuration for the algorithm
	/// \param policy[in]: execution policy
	/// \param first1[in]: iterator to the first element in the first input range
	/// \param last1[in]: iterator to one past the last element in the first input range
	/// \param first2[in]: iterator to the first element in the second input range
	/// \param init[in]: initial value for the reduction
	/// \return result of the transform-reduce operation (inner product)
	template<class ExecPolicy,
			 class RandIt1,
			 class RandIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
				 std::random_access_iterator<RandIt2>
#endif
	/// Performs a parallel transform-reduce operation on two ranges using addition and multiplication
	///
	/// \param policy[in]: execution policy
	/// \param first1[in]: iterator to the first element in the first input range
	/// \param last1[in]: iterator to one past the last element in the first input range
	/// \param first2[in]: iterator to the first element in the second input range
	/// \param init[in]: initial value for the reduction
	/// \return result of the transform-reduce operation (inner product)
	RandIt1::value_type
	transform_reduce(ExecPolicy&& policy,
					 RandIt1 first1,
					 RandIt1 last1,
					 RandIt2 first2,
					 const typename RandIt1::value_type init) noexcept {
		return transform_reduce(std::forward<ExecPolicy>(policy),
			first1, last1, first2, init, std::plus<>(), std::multiplies<>());
	}
}// end namespace cryptanalyslib
#endif //TRANSFORM_H
