#ifndef SMALLSECRETLWE_ALGORITHM_PCS_H
#define SMALLSECRETLWE_ALGORITHM_PCS_H
#include <cstddef>

///
/// \tparam Compare
/// \tparam T
template<class Compare,
		 class T>
class PollardRho {
private:

public:
	constexpr PollardRho() noexcept {};

	/// NOTE: the inputs col1 and col2 are also outputs
	/// \tparam F
	/// \param f
	/// \param col1 input/output the starting value will be set onto this value
	/// \param col2 input/output
	/// \param max_iters: max iters until to exit the algorithm
	/// \return true/false if a solution/collision was found
	template<class F>
	constexpr static bool run(F &&f,
						T &col1, T &col2,
						const size_t max_iters = size_t(-1)) noexcept {
		Compare cmp;

		size_t i = 0;
		T a1=col1, a2, b1=col2, b2;
		while (i < max_iters) {
			a2 = f(a1);
			b2 = f(f(b1));

			if (cmp(a2, b2)) {
				col1 = a1;
				col2 = f(b1);
				return true;
			}

			a1 = a2;
			b1 = b2;
			i += 1;
		}
		return false;
	}
};
#endif
