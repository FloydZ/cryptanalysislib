#ifndef SMALLSECRETLWE_ALGORITHM_PCS_H
#define SMALLSECRETLWE_ALGORITHM_PCS_H
#include <cstddef>


#if __cplusplus > 201709L
/// describes the needed function to be row
template<class C, class T>
concept RhoCompareAble = requires(C c) {
	requires requires (const T a,
	                   const T &aa){
		{ c(aa, aa, aa, aa) } -> std::convertible_to<bool>;
	};
};
#endif



/// \tparam Compare compare object, which needs to take 4
/// \tparam T
template<class Compare,
		 class T>
#if __cplusplus > 201709L
	requires RhoCompareAble<Compare, T>
#endif
class PollardRho {
private:

public:
	constexpr PollardRho() noexcept {};

	/// NOTE: the inputs col1 and col2 are also outputs
	/// \tparam F
	/// \param f lambda function to execute a single step.
	/// \param col1 input/output the starting value will be set onto this value,
	/// 		and if a collision is found in will be written into this value
	///			NOTE: the element before the collision will be return
	/// \param col2 input/output
	/// \param max_iters: maximal iterations until to exit the algorithm
	/// \return true/false if a solution/collision was found
	template<class F>
	[[nodiscard]] constexpr static bool run(F &&f,
						T &col1, T &col2,
						const size_t max_iters = size_t(-1ull)) noexcept {
		Compare cmp;

		size_t i = 0;
		T a1=col1, b1=col2;
		while (i < max_iters) {
			const T a2 = f(a1);
			const T b2_ =f(b1);
			const T b2 = f(b2_);

			if (cmp(a1, a2, b2_, b2)) {
				col1 = a1;
				col2 = b2_;
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
