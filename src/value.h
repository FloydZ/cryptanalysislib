#ifndef SMALLSECRETLWE_VALUE_H
#define SMALLSECRETLWE_VALUE_H

#include <string>       	// for error mngt and printing capabilities.
#include <iostream>
#include <array>            // for the internal data structures
#include <cstdint>

#include "helper.h"
#include "container.h"

#if __cplusplus > 201709L
/// Requirements for a base data container.
/// \tparam Container
template<class Container>
concept ValueAble = requires(Container c) {
	typename Container::DataType;

	// we need to enforce the existence of some fields
	//TODO { Container::LENGTH } -> std::convertible_to<uint32_t>;

	requires requires(const uint32_t i) {
		c[i];
		c.random();
		c.zero();
		c.is_equal(c, i, i);
		c.is_greater(c, i, i);
		c.is_lower(c, i, i);
		Container::add(c, c, c, i, i);
		Container::sub(c, c, c, i, i);
		Container::set(c, c, i, i);
		Container::cmp(c, c, i, i);
		c.neg(i, i);
		c.data();
		c.is_zero();

		c.print_binary(i, i);
		c.print(i, i);
	};

	// we also have to enforce the existence of some constexpr functions.
	{ Container::binary() } -> std::convertible_to<bool>;
	{ Container::size() } -> std::convertible_to<uint32_t>;
	{ Container::limbs() } -> std::convertible_to<uint32_t>;
	{ Container::bytes() } -> std::convertible_to<uint32_t>;
};
#endif

///	the internal data value __MUST__ hold exactly 'length' values, where a value can be a binary or a k-ary type.
///
///	the following class methods __MUST__ be implemented by Container:
///		- zero()	// zero out the internal data array/vector
///		- random() 	// randomize all values within the internal data array.
///		- bool filter(const uint16_t norm=2, const uint64_t k_lower=0, const uint64_t k_higher=length)
///		- friend Value<length> operator+(Value<length> const &c1, Value<length> const &c2)
///		- static bool add(Value &v3, Value const &v1, Value const &v2, const uint16_t norm=2)
///		- VALUE_TYPE& operator [](uint64_t i)
///		- const VALUE_TYPE& operator [](const uint64_t i) const
///		- inline bool is_equal(const Value &obj, const uint64_t level) const
///		- inline bool is_equal(const Value &obj, const uint64_t k_lower, const uint64_t k_upper) const
///		-
///		- Value& operator =(Value &&obj) noexcept
///		- Value& operator =(Value const &obj)
///		- Value& operator +=(Value const &obj
///
///
/// \tparam length number of elements the internal data array/vector __MUST__ be able to hold.
template<class Container>
#if __cplusplus > 201709L
    requires ValueAble<Container>
#endif
class Value_T {
public:
	typedef typename Container::ContainerLimbType ContainerLimbType;
	typedef typename Container::DataType DataType;
	typedef Container ContainerType;

	constexpr static uint32_t LENGTH = Container::LENGTH;

	/// default constructor
	Value_T() noexcept : __data() {this->zero(); }

	/// Copy Constructor
	/// \param a
	Value_T(const Value_T& a) noexcept : __data(a.__data) {}

    /// zero the complete data vector
    void zero() noexcept { __data.zero(); }

    /// set the whole data array on random data.
    void random() noexcept { __data.random(); }

	/// \return true if every coordinate is zero
	bool is_zero() const noexcept {
		return __data.is_zero();
	}

	/// set v3= v1 + v2 iff. \forall r \in \[ k_lower, k_higher ) |v3[r]| < norm.
	/// IMPORTANT: if the 'norm' requirement is not met by all coordinates, e.g. on coordinate is bigger then the norm
	/// the calculations will stop and the resulting 'v3' will __NOT__ be correct.
	/// \param v3 output: v3 = v1+v2
	/// \param v1 input:
	/// \param v2 input:
	/// \param norm max norm:
	///				if norm == -1u, its ignored
	/// 			returns: true if |v3| >= norm
	/// 			returns: else false
	/// \param k_lower	__MUST__ be >= 0 __AND__ < k_higher. All calculations are done including this coordinate.
	/// \param k_higher	__MUST__ be >= 0 __AND__ > k_lower __AND__ <= length. All calculations are done excluding this coordinate.
	/// \return		true if the resulting 'v3' __MUST__ __NOT__ be added to the list.
	///				false else.
    constexpr static inline bool add(Value_T &v3,
	                       			 Value_T const &v1,
	                       			 Value_T const &v2,
	                       			 const uint64_t k_lower=0,
	                       			 const uint64_t k_higher=LENGTH,
	                       			 const uint32_t norm=-1) noexcept {
		return Container::add(v3.__data, v1.__data, v2.__data, k_lower, k_higher, norm);
	}


	/// same as add. Only for subtraction. Note that it does not filter things
	/// \param v3 output. is overwritten
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower coordinate
	/// \param k_higher upper coordinate
	constexpr static inline void sub(Value_T &v3,
	                                 Value_T const &v1,
	                                 Value_T const &v2,
	                				 const uint64_t k_lower=0,
	                                 const uint64_t k_higher=LENGTH) noexcept {
		Container::sub(v3.__data, v1.__data, v2.__data, k_lower, k_higher);
	}


	/// negate all coordinates on: [k_lower, k_upper)
	/// \param k_lower lower coordinate
	/// \param k_upper upper coordinate
	constexpr inline void neg(const uint64_t k_lower=0,
	                          const uint64_t k_upper=LENGTH) noexcept {
		__data.neg(k_lower, k_upper);
	}

	/// i think this does a 3 way comparison.
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower coordinate
	/// \param k_upper upper coordinate
	/// \return v1[k_lower, k_upper) == v2[k_lower, k_upper)
	constexpr inline static bool cmp(Value_T const &v1,
	                                 Value_T const &v2,
	                                 const uint64_t k_lower=0,
	                                 const uint64_t k_upper=LENGTH) noexcept {
		return Container::cmp(v1.__data, v2.__data, k_lower, k_upper);

	}

	/// v1 = v2 between the coordinates [k_lower, k_upper)
	/// \param v1 input
	/// \param v2  input
	/// \param k_lower lower coordinate
	/// \param k_upper upper coordinate
	constexpr inline static void set(Value_T &v1,
	                                 Value_T const &v2,
	                       			 const uint64_t k_lower=0,
	                                 const uint64_t k_upper=LENGTH) {
		Container::set(v1.__data, v2.__data, k_lower, k_upper);
	}

	/// Assignment operator implementing copy assignment
	/// see https://en.cppreference.com/w/cpp/language/operators
    /// \param obj
    /// \return
	Value_T& operator =(Value_T const &obj) noexcept {
		// fast path: do nothing if they are the same.
		if (likely(this != &obj)) {
			__data = obj.__data;
		}

		return *this;
	}

	/// Assignment operator implementing move assignment
	/// Alternative definition: Value& operator =(Value &&obj) = default;
	/// see https://en.cppreference.com/w/cpp/language/move_assignment
	/// \param obj
	/// \return
	Value_T& operator =(Value_T &&obj) noexcept {
		if (this != &obj) { // self-assignment check expected really?
			// move the data
			__data = std::move(obj.__data);
		}

		return *this;
	}

	/// checks whether this == obj on the interval [k_lower, ..., k_upper]
	/// the level of the calling 'list' object.
	/// \param obj     second value to check
	/// \param k_lower lower limit
	/// \param k_upper upper limit
	/// \return true/false
	constexpr inline bool is_equal(const Value_T &obj,
	                               const uint32_t k_lower=0,
	                               const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < k_upper);

		for (uint64_t i = k_lower; i < k_upper; ++i) {
			if (__data[i] != obj.__data[i])
				return false;
		}

		return true;
	}

	/// checks whether this > obj on the the interval [k_lower, ..., k_upper]
	/// \param obj		second value to check
	/// \param k_lower  lower limit
	/// \param k_upper  upper limit
	/// \return true/false
	constexpr inline bool is_greater(const Value_T &obj,
	                                 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < k_upper);
		return __data.is_greater(obj.data(), k_lower, k_upper);
	}

	/// same as "is_greater"
	constexpr inline bool is_lower(const Value_T &obj,
	                               const uint32_t k_lower=0,
	                               const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < k_upper);
		return __data.is_lower(obj.data(), k_lower, k_upper);
	}

	/// \return true/false
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline bool is_equal(const Value_T &obj) const noexcept {
		return __data.template is_equal<k_lower, k_upper>(obj.__data);
	}

	/// \return this > obj between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline bool is_greater(const Value_T &obj) const noexcept {
		return __data.template is_greater<k_lower, k_upper>(obj.__data);
	}

	/// \return this < obj between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_lower(const Value_T &obj) const noexcept {
		return __data.template is_lower<k_lower, k_upper>(obj.__data);
	}

	/// print the data in binary
	void print_binary(const uint64_t k_lower=0, const uint64_t k_upper=LENGTH) const noexcept {
		__data.print_binary(k_lower, k_upper);
	}

	/// print the data
	void print(const uint64_t k_lower=0, const uint64_t k_upper=LENGTH) const noexcept {
		__data.print(k_lower, k_upper);
	}

	///
	/// \return
	__FORCEINLINE__ constexpr static bool binary() noexcept { return Container::binary(); }
	__FORCEINLINE__ constexpr static uint64_t size() noexcept { return Container::size(); }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return Container::limbs(); }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return Container::bytes(); }

	__FORCEINLINE__ Container& data() noexcept {return __data; };
	__FORCEINLINE__ const Container& data() const noexcept { return __data; };

	__FORCEINLINE__ auto data(const uint64_t i) noexcept { ASSERT(i < __data.size()); return __data[i]; };
	__FORCEINLINE__ const DataType data(const uint64_t i) const noexcept { ASSERT(i < __data.size()); return __data[i]; };

	__FORCEINLINE__ auto operator [](const uint64_t i) noexcept { return data(i); };
	__FORCEINLINE__ const DataType operator [](const uint64_t i) const noexcept { return data(i); };
private:
	Container __data;
};

/// \param out out stream
/// \param data input data
/// \return
template<class Container>
std::ostream& operator<< (std::ostream &out, const Value_T<Container> &obj) {
	for (unsigned int i = 0; i < obj.size(); i++){
		out << obj.data()[i] << "";
	}
	return out;
}

#endif //SMALLSECRETLWE_VALUE_H
