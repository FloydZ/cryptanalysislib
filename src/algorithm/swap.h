#pragma once

#include <cstddef>
#include <type_traits>

namespace cryptanalysislib {
	/// Exchanges the values of a and b.
	/// NOTE: the requires-clause makes this overload more constrained than
	/// `std::swap`, so unqualified `swap(a, b)` calls that see both (e.g.
	/// through `using namespace cryptanalysislib;`) are not ambiguous.
	/// \param a[in/out]: first value
	/// \param b[in/out]: second value
	template<typename T>
	    requires std::is_move_constructible_v<T> &&
	             std::is_move_assignable_v<T>
	constexpr void swap(T &a, T &b) noexcept(std::is_nothrow_move_constructible_v<T> &&
	                                         std::is_nothrow_move_assignable_v<T>) {
		T t = static_cast<T &&>(a);
		a = static_cast<T &&>(b);
		b = static_cast<T &&>(t);
	}

	/// Exchanges the arrays a and b element by element.
	/// \param a[in/out]: first array
	/// \param b[in/out]: second array
	template<typename T, const size_t N>
	    requires std::is_move_constructible_v<T> &&
	             std::is_move_assignable_v<T>
	constexpr void swap(T (&a)[N], T (&b)[N]) noexcept(std::is_nothrow_move_constructible_v<T> &&
	                                                   std::is_nothrow_move_assignable_v<T>) {
		for (size_t i = 0; i < N; ++i) {
			cryptanalysislib::swap(a[i], b[i]);
		}
	}

	/// Exchanges the values the two iterators point to.
	/// \param a[in]: iterator to the first value
	/// \param b[in]: iterator to the second value
	template<class ForwardIt1,
	         class ForwardIt2>
	constexpr void iter_swap(ForwardIt1 a,
	                         ForwardIt2 b) {
		cryptanalysislib::swap(*a, *b);
	}

	/// TODO all other functions
} // end namespace cryptanalysislib
