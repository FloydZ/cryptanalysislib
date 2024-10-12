#ifndef CRYPTANALYSISLIB_ALGORITHM_APPLY_H
#define CRYPTANALYSISLIB_ALGORITHM_APPLY_H

#include <tuple>
#include <vector>
#include <future>

namespace cryptanalysislib {
	/// \tparam ExecPolicy
	/// \tparam Op
	/// \tparam ArgContainer
	/// \param policy
	/// \param op
	/// \param args_list
	/// \return
	template <class ExecPolicy,
			  class Op,
			  class ArgContainer>
	std::vector<std::future<void>>
	parallel_apply(ExecPolicy &&policy,
				   Op op,
				   const ArgContainer& args_list) noexcept {
		std::vector<std::future<void>> futures;
		auto& task_pool = *policy.pool();

		for (const auto& args : args_list) {
			futures.emplace_back(task_pool.submit([](Op op, const auto& args_fwd) {
					 std::apply(op, args_fwd);
				 }, op, args));
		}

		return futures;
	}
} // end namespace cryptanalysislib
#endif //APPLY_H
