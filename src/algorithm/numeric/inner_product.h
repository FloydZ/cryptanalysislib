#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_INNER_PRODUCT_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_INNER_PRODUCT_H

#include <iterator>
#include <utility>

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
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_INNER_PRODUCT_H
