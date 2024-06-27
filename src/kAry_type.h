#ifndef SMALLSECRETLWE_KARY_TYPE_H
#define SMALLSECRETLWE_KARY_TYPE_H

#include <cmath>
#include <cstdint>
#include <iostream>

#include "helper.h"
#include "random.h"
#include "math/math.h"
#include "print/print.h"
#include "popcount/popcount.h"

/// \tparam q  modulus
template<const uint64_t q>
class kAry_Type_T {
public:
	using T = TypeTemplate<q>;
	using T2 = TypeTemplate<2*q>;

	static_assert(q > 1);
	static_assert(sizeof(T) < sizeof(T2));

	// we are godd C++ devs
	typedef T ContainerLimbType;
	using DataType = T;
	using ContainerType = kAry_Type_T<q>;

	// list compatibility typedef
	typedef T LimbType;
	typedef T LabelContainerType;

	// make the length and modulus of the container public available
	constexpr static uint32_t LENGTH = bits_log2(q) + 1;
	constexpr static uint32_t MODULUS = q;

private:
	/// computes the comparison mask to compare two values
	/// between the bits [lower, upper)
	/// \param lower lower bound
	/// \param upper upper bound
	/// \return mask
	static constexpr inline const T compute_mask(const uint32_t lower,
	                                      		const uint32_t upper) {
		const T mask1 = (T(1u) << (sizeof(T) - lower)) - T(1u);
		const T mask2 = ~((1u << upper) - T(1u));
		const T mask = mask1 & mask2;
		return mask;
	}

public:
	///
	constexpr kAry_Type_T() noexcept {
		__value = 0;
	};

	///
	/// \param i
	constexpr kAry_Type_T(const uint16_t i) noexcept {
		__value = i % q;
	};

	///
	/// \param in
	constexpr kAry_Type_T(const kAry_Type_T &in) {
		this->__value = in.__value;
	}

	///
	void random() noexcept {
		__value = fastrandombytes_uint64() % q;
	}

	///
	/// \return
	T abs() const noexcept {
		return std::abs(__value);
	}

	///
	/// \param a
	/// \param b
	void addmul(const T &a, const T &b) noexcept {
		__value = (__value + ((T2) a * (T2) b) % q) % q;
	}

	///
	/// \param a
	/// \param b
	void addmul(kAry_Type_T<q> const &a, kAry_Type_T<q> const &b) noexcept {
		__value = T2(__value + T2(T2(a.__value) * T2(b.__value)) % q) % q;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator&(kAry_Type_T< q> obj1, long const obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = (T2(obj1.__value) & T2(obj2)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator^(kAry_Type_T< q> obj1, long const obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = (T2(obj1.__value) ^ T2(obj2)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator*(kAry_Type_T< q> obj1, long const obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = T((T2(T2(obj1.__value) * T2(obj2))) % q);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator&(const kAry_Type_T< q> obj1, kAry_Type_T< q> const &obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = (T2(obj1.__value) & T2(obj2.__value)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator^(kAry_Type_T< q> obj1, kAry_Type_T< q> const &obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = (T2(obj1.__value) ^ T2(obj2.__value)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator*(kAry_Type_T< q> obj1, kAry_Type_T< q> const &obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = (T2(T2(obj1.__value) * T2(obj2.__value))) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T< q> operator+(kAry_Type_T< q> obj1, kAry_Type_T< q> const &obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = (T2(T2(obj1.__value) + T2(obj2.__value))) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T< q> operator-(kAry_Type_T< q> obj1, kAry_Type_T< q> const &obj2) noexcept {
		kAry_Type_T< q> r;
		r.__value = obj1.__value >= obj2.__value ? T2(obj1.__value) - T2(obj2.__value) : obj1.__value + (q - obj2.__value);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend bool operator==(uint64_t obj1, kAry_Type_T< q> const &obj2) noexcept {
		return obj1 == obj2.__value;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend bool operator!=(uint64_t obj1, kAry_Type_T< q> const &obj2) noexcept {
		return obj1 != obj2.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator+=(unsigned int obj) noexcept {
		__value = T2((T2) __value + (T2) obj) % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator+=(kAry_Type_T< q> const &obj) noexcept {
		__value = T2((T2) __value + (T2) obj.__value) % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator%=(const uint64_t &obj) noexcept {
		__value %= obj;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator%(const uint64_t &obj) noexcept {
		__value %= obj;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator=(kAry_Type_T< q> const &obj) noexcept {
		if (this != &obj) {
			__value = obj.__value;
		}

		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator=(uint32_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator=(uint64_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator=(int32_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator=(int64_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	//	kAry_Type_T< q> &operator=(T const obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	//	kAry_Type_T< q> &operator=(unsigned int const obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	//	kAry_Type_T< q> &operator=(unsigned long obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator!=(kAry_Type_T< q> const &obj) const noexcept {
		return __value != obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator==(kAry_Type_T< q> const &obj) const noexcept {
		return __value == obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator==(uint64_t const &obj) const noexcept {
		return __value == obj;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator>(kAry_Type_T< q> const &obj) const noexcept {
		return __value > obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator>=(kAry_Type_T< q> const &obj) const noexcept {
		return __value >= obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator<(kAry_Type_T< q> const &obj) const noexcept {
		return __value < obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator<=(kAry_Type_T< q> const &obj) const noexcept {
		return __value <= obj.__value;
	}

	///
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	static constexpr inline bool cmp(kAry_Type_T const &o1,
	                                 kAry_Type_T const &o2,
								   const uint32_t lower=0,
								   const uint32_t upper=LENGTH) noexcept {
		return o1.is_equal(o2, lower, upper);
	}

	///
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_equal(kAry_Type_T const &o,
								   const uint32_t lower=0,
								   const uint32_t upper=LENGTH) const noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);

		const T mask = compute_mask(lower, upper);
		return (__value & mask) == (o.value() & mask);
	}

	///
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_greater(kAry_Type_T const &o,
								     const uint32_t lower=0,
								     const uint32_t upper=LENGTH) const noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);

		const T mask = compute_mask(lower, upper);
		return (__value & mask) > (o.value() & mask);
	}

	///
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_lower(kAry_Type_T const &o,
	                               const uint32_t lower=0,
	                               const uint32_t upper=LENGTH) const noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);

		const T mask = compute_mask(lower, upper);
		return (__value & mask) < (o.value() & mask);
	}

	///
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_zero(const uint32_t lower=0,
	                              const uint32_t upper=LENGTH) const noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		return (__value & mask) == T(0);
	}

	/// not really useful, if lower != 0 and upper != lengh
	constexpr inline static void add(kAry_Type_T &out,
									 kAry_Type_T &in1,
									 kAry_Type_T &in2,
									 const uint32_t lower=0,
									 const uint32_t upper=LENGTH) noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		const T tmp = (in1 & mask) + (in2 & mask);
		out += tmp;
	}

	/// not really useful, if lower != 0 and upper != lengh
	constexpr inline static void sub(kAry_Type_T &out,
	                                 kAry_Type_T &in1,
	                                 kAry_Type_T &in2,
	        						 const uint32_t lower=0,
	                                 const uint32_t upper=LENGTH) noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		const T tmp = (in1 & mask) - (in2 & mask); // TODO not really correct
		out -= tmp;
	}

	///
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline T neg(const uint32_t lower=0,
	                       const uint32_t upper=LENGTH) const noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		return __value ^ mask;
	}

	static constexpr inline LimbType add_T(const LimbType a, const LimbType b) noexcept {
		return (a + b) % q;
	}

	static constexpr inline LimbType sub_T(const LimbType a, const LimbType b) noexcept {
		return (q + a - b) % q;
	}

	static constexpr inline LimbType mul_T(const LimbType a, const LimbType b) noexcept {
		return (T2(a) - T2(b)) % T2(q);
	}

	static constexpr inline LimbType scalar_T(const LimbType a, const LimbType b) noexcept {
		return (T2(a) - T2(b)) % T2(q);
	}

	static constexpr inline LimbType mod_T(const LimbType a) noexcept {
		return a % q;
	}

	static constexpr inline LimbType neg_T(const LimbType a) noexcept {
		return (q - a) % q;
	}

	static constexpr inline LimbType popcnt_T(const LimbType a) noexcept {
		return cryptanalysislib::popcount::popcount(a);
	}


	T get_value() noexcept { return __value; }
	T value() noexcept { return __value; }

	const T get_value() const noexcept { return __value; }
	const T value() const noexcept { return __value; }
	const T data() const noexcept { return __value; }
	constexpr uint32_t bits() const noexcept { return LENGTH; }
	constexpr static T modulo() noexcept { return q; }

	/// not really meaningfully, but needed by the API:
	/// \param i
	/// \return returns the value
	T operator[](const size_t i) noexcept {
		ASSERT(i < LENGTH);
		return value;
	}

	/// also not really meaningfully
	/// \param i
	/// \return
	constexpr inline T get(const size_t i) noexcept {
		ASSERT(i < LENGTH);
		return value;
	}

	/// sets out = in[lower, upper)
	/// \param out
	/// \param in
	/// \param lower
	/// \param upper
	/// \return
	constexpr static inline void set(kAry_Type_T &out,
	                                 const kAry_Type_T &in,
	                                 const size_t lower,
	                                 const size_t upper) noexcept {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		const T tmp = in.value() & mask;
		out = (out & ~mask) ^ tmp;
	}

	/// NOTE: not really useful; i is ignored
	/// \param val
	/// \param i
	/// \return
	constexpr inline void set(const T val, const size_t i) noexcept {
		ASSERT(i < LENGTH);
		__value = val;
	}

	///
	/// \return
	constexpr inline void zero() noexcept {
		__value = 0;
	}

	///
	/// \return
	constexpr inline T* ptr() noexcept {
		return &__value;
	}

	/// NOTE: not really useful: i is ignored
	/// \param i
	/// \return
	constexpr inline T* ptr(const size_t i) noexcept {
		ASSERT(i < LENGTH);
		return __value;
	}

	constexpr void print_binary(const uint32_t lower,
	                            const uint32_t upper) {
		ASSERT(sizeof(T) > lower);
		ASSERT(sizeof(T) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		const T tmp = (__value & mask) >> lower;
		cryptanalysislib::print_binary(tmp);
	}

	/// NOTE: lower and upper are ignored
	constexpr void print(const uint32_t lower,
	                     const uint32_t upper) {
		(void) lower;
		(void) upper;
		std::cout << __value << std::endl;
	}

	static constexpr inline bool binary() noexcept {
		return q == 2;
	}

	/// returns the number of elements stored this container
	static constexpr inline size_t size() noexcept {
		return 1;
	}

	/// returns the number of elements stored this container
	static constexpr inline size_t limbs() noexcept {
		return 1;
	}

	static constexpr inline size_t bytes() noexcept {
		return sizeof(T);
	}
private:
	T __value;
};

template<typename T, typename T2, const T q>
std::ostream &operator<<(std::ostream &out, const kAry_Type_T< q> &obj) {
	out << signed(obj.get_value());
	return out;
}

//generic abs function for the kAryType
template<class T>
T abs(T in) {
	if constexpr (std::is_integral<T>::value == true) {
		return __builtin_llabs(in);
	} else {
		return T(__builtin_llabs(in.data()));
	}
}

#include "helper.h"
#endif//SMALLSECRETLWE_KARY_TYPE_H
