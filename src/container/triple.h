#ifndef SMALLSECRETLWE_TRIPLE_H
#define SMALLSECRETLWE_TRIPLE_H

#include <type_traits>

// TODO:
//  - implement alignment  options
//  - compare operators

struct ConfigTriple {
	const bool swap12 = false;
	const bool swap23 = false;
	const bool swap13 = false;

	// default constructor
	constexpr ConfigTriple() : swap12(false), swap23(false), swap13(false) {}
};


/// Simple extension of the std::pair data type. Probably one could have used `std::tuple`, but
/// i want to experiment a little bit with automatic alignment optimizations.
/// \tparam T1
/// \tparam T2
/// \tparam T3
template<typename T1, typename T2, typename T3, const ConfigTriple &config>
	requires
		std::is_default_constructible_v<T1> &&
		std::is_default_constructible_v<T2> &&
		std::is_default_constructible_v<T3>
struct  __attribute__ ((packed))
    triple {
public:
	using first_type    = typename std::conditional<config.swap12, T2, T1>::type;
	using second_type   = typename std::conditional<config.swap23, T3, T2>::type;
	using third_type    = typename std::conditional<config.swap13, T1, T3>::type;

	/// automatic alignment deduction.
	first_type first;
	second_type second;
	third_type third;

	/// simple constructor.
	constexpr triple() noexcept :
		first(), second(), third() {}

	/// simple constructor, default initializes the fields T1
	constexpr triple(const T1 &a) noexcept :
		first(a), second(), third() {}

	/// simple constructor, default initializes the fields T1 and T2
	constexpr triple(const T1 &a, const T2 &b) noexcept :
		first(a), second(b), third() {}

	/// simple constructor, default initializes everything
	constexpr triple(const T1 &a, const T2 &b, const T3 &c) noexcept :
		first(a), second(b), third(c) {}

	/// copy constructor
	//constexpr triple(const triple&) = default;

	/// move constructor
	//constexpr triple(triple&&) = default;

	constexpr auto get_first() noexcept {
		if constexpr(config.swap12) {
			return second;
		}

		return false;
	}

	constexpr void swap(const triple &t) noexcept {
		std::swap(first, t.first);
		std::swap(second, t.second);
		std::swap(third, t.third);
	}
};

#endif //SMALLSECRETLWE_TRIPLE_H
