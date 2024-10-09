#ifndef CRYPTANALYSISLIB_ALGORITHM_FOR_EACH_H
#define CRYPTANALYSISLIB_ALGORITHM_FOR_EACH_H

#ifndef __APPLE__

#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {

	/// NOTE: Iterators are expected to be rng access.
    /// See std::for_each https://en.cppreference.com/w/cpp/algorithm/for_each
    /// \tparam ExecutionPolicy
    /// \tparam RandIt
    /// \tparam UnaryFunction
    /// \param policy
    /// \param first
    /// \param last
    /// \param f
    /// \return
	template <class ExecutionPolicy,
	          class RandIt,
	          class UnaryFunction>
#if __cplusplus > 201709L
		requires std::is_execution_policy_v<std::remove_cvref_t<std::remove_cvref_t<ExecutionPolicy>>> &&
		         std::random_access_iterator<RandIt>
#endif
	void for_each(ExecutionPolicy &&policy,
	              RandIt first,
	              RandIt last,
	              UnaryFunction f) noexcept {
		// TODO double check
		// Using a lambda instead of just calling the
		// non-policy std::for_each because it appears to
		// result in a smaller binary.
		auto chunk_func = [&f](RandIt chunk_first, RandIt chunk_last) __attribute__((always_inline)) {
			for (; chunk_first != chunk_last; ++chunk_first) {
				f(*chunk_first);
			}
		};

		if (not policy.__allow_parallel()) {
			chunk_func(first, last);
			return;
		}

		internal::parallel_chunk_for_1_wait(std::forward<ExecutionPolicy>(policy), first, last,
		                                             chunk_func, (void*)nullptr, 1);
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam Size
	/// \tparam UnaryFunction
	/// \param policy
	/// \param first
	/// \param n
	/// \param f
	/// \return
	template <class ExecPolicy,
			  class RandIt,
			  class Size,
			  class UnaryFunction>
    RandIt for_each_n(ExecPolicy &&policy, RandIt first, Size n, UnaryFunction f) {
        RandIt last = internal::advanced(first, n);
        cryptanalysislib::for_each(std::forward<ExecPolicy>(policy), first, last, f);
        return last;
    }
}
#endif // `__APPLE__
#endif
