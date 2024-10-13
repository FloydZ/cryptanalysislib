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

	/// \tparam ForwardIt1
	/// \tparam ForwardIt2
	/// \tparam BinaryOp1
	/// \tparam BinaryOp2
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \param reduce
	/// \param transform
	/// \return
	template<class ForwardIt1,
			 class ForwardIt2,
			 class BinaryOp1,
			 class BinaryOp2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
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


	/// \tparam InputIt1
	/// \tparam InputIt2
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \return
	template<class InputIt1,
			 class InputIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
	InputIt1::value_type transform_reduce(InputIt1 first1,
										  InputIt1 last1,
										  InputIt2 first2,
										  const typename InputIt1::value_type init) noexcept {
		using T = InputIt1::value;
		return transform_reduce(first1, last1, first2, init, std::plus<T>(), std::multiplies<T>());
	}


	/// \tparam InputIt
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

	/// @tparam ExecPolicy
	/// @tparam RandIt1
	/// @tparam BinaryReductionOp
	/// @tparam UnaryTransformOp
	/// @tparam config
	/// @param policy
	/// @param first1
	/// @param last1
	/// @param init
	/// @param reduce_op
	/// @param transform_op
	/// @return
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
		using T = RandIt1::value_type;
		const auto size = static_cast<size_t>(std::distance(first1, last1));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return std::transform_reduce(first1, last1, init, reduce_op, transform_op);
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

	template <class ExecPolicy,
			  class RandIt1,
			  class RandIt2,
			  class BinaryReductionOp,
			  class BinaryTransformOp,
			  const AlgorithmTransformConfig &config=algorithmTransformConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
				 std::random_access_iterator<RandIt2>
#endif
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
				 std::transform_reduce<RandIt1, RandIt2, T, BinaryReductionOp, BinaryTransformOp>,
				(T*)nullptr,
				nthreads,
				init, reduce_op, transform_op);

		return std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), init, reduce_op);
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt1
	/// \tparam RandIt2
	/// \tparam config
	/// \param policy
	/// \param first1
	/// \param last1
	/// \param first2
	/// \param init
	/// \return
	template<class ExecPolicy,
			 class RandIt1,
			 class RandIt2,
			 const AlgorithmTransformConfig &config=algorithmTransformConfig>
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
