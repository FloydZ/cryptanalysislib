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

	/// \tparam InputIt
	/// \tparam BinaryOp
	/// \param first
	/// \param last
	/// \param init
	/// \param op
	/// \return
	template<class InputIt,
			 class BinaryOp,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt>
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

	/// \tparam InputIt
	/// \param first
	/// \param last
	/// \param init
	/// \return
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

	/// \tparam InputIt
	/// \param first
	/// \param last
	/// \return
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

	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam UnaryOperation
	/// \tparam config
	/// \param first1
	/// \param last1
	/// \param d_first
	/// \param unary_op
	/// \return
	template<class InputIt,
			 class OutputIt,
			 class UnaryOperation,
			 const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::forward_iterator<OutputIt> &&
    			 std::regular_invocable<UnaryOperation,
										const typename InputIt::value_type&>
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

	/// \tparam InputIt1
	/// \tparam InputIt2
	/// \tparam OutputIt
	/// \tparam BinaryOperation
	/// \tparam config
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param d_first
	/// \param binary_op
	/// \return
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
										const typename InputIt1::value_type&>
#endif
    OutputIt transform(InputIt1 first1, InputIt1 last1,
                       InputIt2 first2, OutputIt d_first,
                       UnaryOperation binary_op) noexcept {
        while (first1 != last1) {
            *d_first++ = binary_op(*first1++, *first2++);
        }

        return d_first;
    }

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam UnaryOperation
	/// \tparam config
	/// \param policy
	/// \param first
	/// \param last
	/// \param init
	/// \param binop
	/// \return
	template <class ExecPolicy,
			  class RandIt,
			  class UnaryOperation,
			  const AlgorithmReduceConfig &config=algorithmReduceConfig>
#if __cplusplus > 201709L
	    requires std::random_access_iterator<RandIt> &&
    			 std::regular_invocable<UnaryOperation,
										const typename RandIt::value_type&>
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

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam config
	/// \param policy
	/// \param first
	/// \param last
	/// \param init
	/// \return
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
		return cryptanalysislib::reduce(std::forward<ExecPolicy>(policy),
						   first, last, init, std::plus<T>());
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam config
	/// \param policy
	/// \param first
	/// \param last
	/// \return
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
