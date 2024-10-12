#ifndef CRYPTANALYSISLIB_ALGORITHM_TRANSFORM_H
#define CRYPTANALYSISLIB_ALGORITHM_TRANSFORM_H

#include <numeric>

#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {

	struct AlgorithmTransformConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 10u;
	};
	constexpr static AlgorithmTransformConfig algorithmTransformConfig;

	/// @tparam InputIt
	/// @tparam OutputIt
	/// @tparam UnaryOp
	/// @param first1
	/// @param last1
	/// @param d_first
	/// @param unary_op
	/// @return
	template<class InputIt,
			 class OutputIt,
			 class UnaryOp,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt> &&
				 std::random_access_iterator<OutputIt>
#endif
	constexpr OutputIt transform(InputIt first1,
								 InputIt last1,
								 OutputIt d_first,
								 UnaryOp unary_op) noexcept {
		for (; first1 != last1; ++d_first, ++first1) {
			*d_first = unary_op(*first1);
		}

		return d_first;
	}

	/// \tparam InputIt1
	/// \tparam InputIt2
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param d_first
	/// \param binary_op
	/// \return
	template<class InputIt1,
			 class InputIt2,
			 class OutputIt,
			 class BinaryOp,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt1> &&
				 std::random_access_iterator<InputIt2>
#endif
	constexpr OutputIt transform(InputIt1 first1,
								 InputIt1 last1,
								 InputIt2 first2,
								 OutputIt d_first,
								 BinaryOp binary_op) noexcept {
		for (; first1 != last1; ++d_first, ++first1, ++first2) {
			*d_first = binary_op(*first1, *first2);
		}

		return d_first;
	}

	/// \tparam InputIt1
	/// \tparam InputIt2
	/// \tparam T
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \return
	template<class InputIt1,
			 class InputIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	T transform_reduce( InputIt1 first1, InputIt1 last1,
                    InputIt2 first2, T init );

	/// \tparam ExecutionPolicy
	/// \tparam ForwardIt1
	/// \tparam ForwardIt2
	/// \tparam T
	/// \param policy
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \return
	template<class ExecutionPolicy,
			 class ForwardIt1,
			 class ForwardIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	T transform_reduce( ExecutionPolicy&& policy,
						ForwardIt1 first1, ForwardIt1 last1,
						ForwardIt2 first2, T init );

	/// \tparam InputIt1
	/// \tparam InputIt2
	/// \tparam T
	/// \tparam BinaryOp1
	/// \tparam BinaryOp2
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \param reduce
	/// \param transform
	/// \return
	template<class InputIt1,
			 class InputIt2,
			 class BinaryOp1, class BinaryOp2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	T transform_reduce( InputIt1 first1, InputIt1 last1,
						InputIt2 first2, T init,
						BinaryOp1 reduce, BinaryOp2 transform );

	/// \tparam ExecutionPolicy
	/// \tparam ForwardIt1
	/// \tparam ForwardIt2
	/// \tparam T
	/// \tparam BinaryOp1
	/// \tparam BinaryOp2
	/// \param policy
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \param reduce
	/// \param transform
	/// \return
	template<class ExecutionPolicy,
			 class ForwardIt1,
			 class ForwardIt2,
			 class BinaryOp1,
			 class BinaryOp2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	T transform_reduce( ExecutionPolicy&& policy,
                    ForwardIt1 first1, ForwardIt1 last1,
                    ForwardIt2 first2, T init,
                    BinaryOp1 reduce, BinaryOp2 transform );

	/// \tparam InputIt
	/// \tparam T
	/// \tparam BinaryOp
	/// \tparam UnaryOp
	/// \param first
	/// \param last
	/// \param init
	/// \param reduce
	/// \param transform
	/// \return
	template<class InputIt,
             class BinaryOp,
			 class UnaryOp,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	T transform_reduce( InputIt first, InputIt last, T init,
                    BinaryOp reduce, UnaryOp transform );

	/// \tparam ExecutionPolicy
	/// \tparam ForwardIt
	/// \tparam T
	/// \tparam BinaryOp
	/// \tparam UnaryOp
	/// \param policy
	/// \param first
	/// \param last
	/// \param init
	/// \param reduce
	/// \param transform
	/// \return
	template<class ExecutionPolicy,
			 class ForwardIt,
			 class T,
			 class BinaryOp,
		     class UnaryOp,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	T transform_reduce( ExecutionPolicy&& policy,
                    ForwardIt first, ForwardIt last, T init,
                    BinaryOp reduce, UnaryOp transform );


	template <class ExecPolicy,
			  class RandIt1,
			  class BinaryReductionOp,
			  class UnaryTransformOp,
			  const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1>
#endif
	typename RandIt1::value_value
	transform_reduce(ExecPolicy&& policy,
					 RandIt1 first1,
					 RandIt1 last1,
					 const typename RandIt1::value_value init,
					 BinaryReductionOp reduce_op,
					 UnaryTransformOp transform_op) noexcept {
		const auto size = static_cast<size_t>(std::distance(first1, last1));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return std::transform_reduce(first1, last1, init, reduce_op, transform_op);
		}

		if (poolstl::internal::is_seq<ExecPolicy>(policy)) {
		}

		auto futures = poolstl::internal::parallel_chunk_for_1(std::forward<ExecPolicy>(policy), first1, last1,
															   std::transform_reduce<RandIt1, T,
																				   BinaryReductionOp, UnaryTransformOp>,
															   (T*)nullptr, 1, init, reduce_op, transform_op);

		return poolstl::internal::cpp17::reduce(
			poolstl::internal::get_wrap(futures.begin()),
			poolstl::internal::get_wrap(futures.end()), init, reduce_op);
	}

	template <class ExecPolicy, class RandIt1, class RandIt2, class T, class BinaryReductionOp, class BinaryTransformOp,
			  const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
				 std::random_access_iterator<RandIt2>
#endif
	poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
	transform_reduce(ExecPolicy&& policy, RandIt1 first1, RandIt1 last1, RandIt2 first2, T init,
					 BinaryReductionOp reduce_op, BinaryTransformOp transform_op) {
		if (poolstl::internal::is_seq<ExecPolicy>(policy)) {
			return std::transform_reduce(first1, last1, first2, init, reduce_op, transform_op);
		}

		auto futures = poolstl::internal::parallel_chunk_for_2(std::forward<ExecPolicy>(policy), first1, last1, first2,
															   std::transform_reduce<RandIt1, RandIt2, T,
																				  BinaryReductionOp, BinaryTransformOp>,
															   (T*)nullptr, init, reduce_op, transform_op);

		return poolstl::internal::cpp17::reduce(
			poolstl::internal::get_wrap(futures.begin()),
			poolstl::internal::get_wrap(futures.end()), init, reduce_op);
	}

	template<class ExecPolicy,
			 class RandIt1,
			 class RandIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	poolstl::internal::enable_if_poolstl_policy<ExecPolicy, T>
	transform_reduce(ExecPolicy&& policy, RandIt1 first1, RandIt1 last1, RandIt2 first2, T init ) {
		return transform_reduce(std::forward<ExecPolicy>(policy),
			first1, last1, first2, init, std::plus<>(), std::multiplies<>());
	}
}// end namespace cryptanalyslib
#endif //TRANSFORM_H
