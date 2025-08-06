#ifndef CRYPTANALYSLISLIB_ALGORITHM_REDUCE_H
#define CRYPTANALYSLISLIB_ALGORITHM_REDUCE_H

#include <numeric>

#include "algorithm/algorithm.h"
#include "thread/thread.h"


namespace cryptanalysislib {
	struct AlgorithmReduceConfig : public AlgorithmConfig {
		const size_t min_size_per_thread = 1u << 14u;
	};
	constexpr static AlgorithmReduceConfig algorithmReduceConfig;

	/// Reduces a range of elements using a binary operation
	/// 
	/// \tparam InputIt type of the input iterator
	/// \tparam BinaryOp type of the binary operation
	/// \param first [in]: iterator to the first element
	/// \param last [in]: iterator to one past the last element
	/// \param init [in]: initial value for the reduction
	/// \param op [in]: binary operation function object
	/// \return result of the reduction operation
	template<class InputIt,
			 class BinaryOp,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
    			 std::regular_invocable<BinaryOp,
										typename InputIt::value_type&,
										typename InputIt::value_type&>
#endif
	constexpr typename InputIt::value_type reduce(InputIt first,
												  InputIt last,
												  const typename InputIt::value_type init,
												  BinaryOp op) noexcept {
		using T = InputIt::value_type;
		if (first == last) { return T{}; }

		T ret = init;
		for (; first != last; ++first) {
			ret = op(ret, *first);
		}

		return ret;

	}

	/// Reduces a range of elements using addition
	/// 
	/// \tparam InputIt type of the input iterator
	/// \param first [in]: iterator to the first element
	/// \param last [in]: iterator to one past the last element
	/// \param init [in]: initial value for the reduction
	/// \return result of the reduction operation
	template<class InputIt,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt>
#endif
	constexpr typename InputIt::value_type reduce(InputIt first,
												  InputIt last,
												  const typename InputIt::value_type init) noexcept {
		using T = InputIt::value_type;
		if (first == last) { return T{}; }

		auto op = std::plus<T>();

		T ret = init;
		for (; first != last; ++first) {
			ret = op(ret, *first);
		}

		return ret;
	}

	/// Reduces a range of elements using addition with initial value of 0
	/// 
	/// \tparam InputIt type of the input iterator
	/// \param first [in]: iterator to the first element
	/// \param last [in]: iterator to one past the last element
	/// \return result of the reduction operation
	template<class InputIt,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt>
#endif
	constexpr typename InputIt::value_type reduce(InputIt first,
												  InputIt last) noexcept {
		using T = InputIt::value_type;
		return cryptanalysislib::reduce(first, last, (T)0);
	}

	/// Applies a unary operation to a range of elements and stores the result in another range
	/// 
	/// \tparam InputIt type of the input iterator
	/// \tparam OutputIt type of the output iterator
	/// \tparam UnaryOperation type of the unary operation
	/// \tparam config configuration for the algorithm
	/// \param first1 [in]: iterator to the first element in the input range
	/// \param last1 [in]: iterator to one past the last element in the input range
	/// \param d_first [out]: iterator to the first element in the output range
	/// \param unary_op [in]: unary operation function object
	/// \return iterator to the element past the last element written
	template<class InputIt,
			 class OutputIt,
			 class UnaryOperation,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::forward_iterator<OutputIt> &&
    			 std::regular_invocable<UnaryOperation,
										typename InputIt::value_type&>
#endif
    OutputIt transform(InputIt first1,
    				   InputIt last1,
    				   OutputIt d_first,
                       UnaryOperation unary_op) noexcept {
        while (first1 != last1) {
            *d_first++ = unary_op(*first1++);
        }

        return d_first;
    }

	/// Applies a binary operation to pairs of elements from two ranges and stores the result in a third range
	/// 
	/// \tparam InputIt1 type of the first input iterator
	/// \tparam InputIt2 type of the second input iterator
	/// \tparam OutputIt type of the output iterator
	/// \tparam UnaryOperation type of the binary operation
	/// \tparam config configuration for the algorithm
	/// \param first1 [in]: iterator to the first element in the first input range
	/// \param last1 [in]: iterator to one past the last element in the first input range
	/// \param first2 [in]: iterator to the first element in the second input range
	/// \param d_first [out]: iterator to the first element in the output range
	/// \param binary_op [in]: binary operation function object
	/// \return iterator to the element past the last element written
	template<class InputIt1,
			 class InputIt2,
			 class OutputIt,
			 class UnaryOperation,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt1> &&
	    		 std::forward_iterator<InputIt2> &&
				 std::forward_iterator<OutputIt> &&
    			 std::regular_invocable<UnaryOperation,
										typename InputIt1::value_type&,
										typename InputIt1::value_type&>
#endif
    OutputIt transform(InputIt1 first1, InputIt1 last1,
                       InputIt2 first2, OutputIt d_first,
                       UnaryOperation binary_op) noexcept {
        while (first1 != last1) {
            *d_first++ = binary_op(*first1++, *first2++);
        }

        return d_first;
    }

	/// Parallel version of reduce that uses an execution policy
	/// 
	/// \tparam ExecPolicy type of the execution policy
	/// \tparam RandIt type of the random access iterator
	/// \tparam UnaryOperation type of the binary operation
	/// \tparam config configuration for the algorithm
	/// \param policy [in]: execution policy
	/// \param first [in]: iterator to the first element
	/// \param last [in]: iterator to one past the last element
	/// \param init [in]: initial value for the reduction
	/// \param binop [in]: binary operation function object
	/// \return result of the reduction operation
	template <class ExecPolicy,
			  class RandIt,
			  class UnaryOperation,
			  const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::random_access_iterator<RandIt> &&
    			 std::regular_invocable<UnaryOperation,
										typename RandIt::value_type&,
										typename RandIt::value_type&>
#endif
	typename RandIt::value_type
	reduce(ExecPolicy &&policy,
		   RandIt first,
		   RandIt last,
		   const typename RandIt::value_type init,
		   UnaryOperation binop) noexcept {
		using T = RandIt::value_type;
		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::reduce(first, last, init, binop);
		}

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy), first, last,
					cryptanalysislib::reduce<RandIt, UnaryOperation, config>,
					(T*)nullptr,
					1,
					nthreads,
					init, binop);

		return std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), init, binop);
	}

	/// Parallel version of reduce using addition that uses an execution policy
	/// 
	/// \tparam ExecPolicy type of the execution policy
	/// \tparam RandIt type of the random access iterator
	/// \tparam config configuration for the algorithm
	/// \param policy [in]: execution policy
	/// \param first [in]: iterator to the first element
	/// \param last [in]: iterator to one past the last element
	/// \param init [in]: initial value for the reduction
	/// \return result of the reduction operation
	template <class ExecPolicy,
			  class RandIt,
			  const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::random_access_iterator<RandIt>
#endif
	RandIt::value_type
	reduce(ExecPolicy &&policy,
		   RandIt first,
		   RandIt last,
		   const typename RandIt::value_type init) noexcept {
		using T = RandIt::value_type;
		return cryptanalysislib::reduce
			<ExecPolicy, RandIt, decltype(std::plus<T>()), config>
			(std::forward<ExecPolicy>(policy), first, last, init, std::plus<T>());
	}

	/// Parallel version of reduce using addition with default initial value that uses an execution policy
	/// 
	/// \tparam ExecPolicy type of the execution policy
	/// \tparam RandIt type of the random access iterator
	/// \tparam config configuration for the algorithm
	/// \param policy [in]: execution policy
	/// \param first [in]: iterator to the first element
	/// \param last [in]: iterator to one past the last element
	/// \return result of the reduction operation
	template <class ExecPolicy,
			  class RandIt,
			  const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::random_access_iterator<RandIt>
#endif
	RandIt::value_type
	reduce(ExecPolicy &&policy,
		   RandIt first,
		   RandIt last) noexcept {
		return cryptanalysislib::reduce
			<ExecPolicy, RandIt, config>
		(std::forward<ExecPolicy>(policy), first, last,
		typename std::iterator_traits<RandIt>::value_type{});
	}

}// end namespace cryptanalyslib
#endif //REDUCE_H
