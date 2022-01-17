#ifndef SMALLSECRETLWE_TRIPLE_H
#define SMALLSECRETLWE_TRIPLE_H

#include <type_traits>

// TODO:
//  - implement alignment  options
//  - compare opertors

/// Simple extension of the std::pair data type. Probably one could have used `std::tuple`, but
/// i want to experiment a little bit with automatic alignment optimizations.
/// \tparam T1
/// \tparam T2
/// \tparam T3
template<typename T1, typename T2, typename T3>
	requires
		std::is_default_constructible_v<T1> &&
		std::is_default_constructible_v<T2> &&
		std::is_default_constructible_v<T3>
struct  __attribute__ ((packed))
    triple {
public:
	typedef T1 first_type;
	typedef T2 second_type;
	typedef T3 third_type;

	/// automatic alignment deduction.
	T1 first;
	T2 second;
	T3 third;

	/// simple constructor.
	constexpr triple() :
		first(), second(), third() {}

	/// simple constructor, default initializes the fields T2 and T3
	constexpr triple(const T1 &a) :
		first(a), second(), third() {}

	/// simple constructor, default initializes the fields T3
	constexpr triple(const T1 &a, const T2 &b) :
		first(a), second(b), third() {}

	/// simple constructor
	constexpr triple(const T1 &a, const T2 &b, const T3 &c) :
		first(a), second(b), third(c) {}

	/// copy constructor
	constexpr triple(const triple&) = default;
	/// move constructor
	constexpr triple(triple&&) = default;

	constexpr void swap(const triple &t) noexcept {
		std::swap(first, t.first);
		std::swap(second, t.second);
		std::swap(third, t.third);
	}

	/// default copy constructo
	triple& operator =(triple const &obj) = default;
};

#endif //SMALLSECRETLWE_TRIPLE_H
