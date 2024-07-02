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
#include "simd//simd.h"
#include "metric.h"

using namespace cryptanalysislib::metric;

/// \tparam q  modulus
template<const uint64_t _q, 
		 class Metric=HammingMetric>
class kAry_Type_T {
public:

	// make the length and modulus of the container public available
	constexpr static uint64_t q = _q;
	constexpr static inline uint64_t modulus() noexcept { return _q; }
	constexpr static uint64_t n = 1;
	constexpr static inline uint64_t length() noexcept { return 1; }
	constexpr static inline uint64_t bits() noexcept { return bits_log2(q) + 1; }

	using T = TypeTemplate<q>;
	using T2 = TypeTemplate<q*q>;

	static_assert(q > 1);
	// this is needed to make sure that we have enough `bits` in reserve to 
	// correclty compute the multiplication.
	static_assert(sizeof(T) <= sizeof(T2));

	// we are godd C++ devs
	typedef T ContainerLimbType;
	using DataType = T;
	using ContainerType = kAry_Type_T<q>;

	// list compatibility typedef
	typedef T LimbType;
	typedef T LabelContainerType;

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
	constexpr kAry_Type_T(const kAry_Type_T &in) noexcept {
		this->__value = in.__value;
	}

	///
	static inline kAry_Type_T<q> random() noexcept {
		kAry_Type_T ret;
		// Bla Bla not uniform, I dont care
		ret.__value = fastrandombytes_uint64() % q;
		return ret;
	}

	///
	/// \return
	constexpr T abs() const noexcept {
		return std::abs(__value);
	}

	///	return (this * a) + b
	/// \param a
	/// \param b
	constexpr void addmul(const T &a, const T &b) noexcept {
		__value = (__value + ((T2) a * (T2) b) % q) % q;
	}

	///
	/// \param a
	/// \param b
	constexpr void addmul(kAry_Type_T<q> const &a, kAry_Type_T<q> const &b) noexcept {
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
	friend kAry_Type_T operator-(kAry_Type_T obj1, kAry_Type_T const &obj2) noexcept {
		kAry_Type_T r;
		r.__value = T((T2(T2(obj1.__value)  + T2(q) - T2(obj2.__value))) % T2(q));
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
	constexpr kAry_Type_T< q> &operator+=(kAry_Type_T const &obj) noexcept {
		__value = T2((T2) __value + (T2) obj.__value) % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T< q> &operator-=(kAry_Type_T const &obj) noexcept {
		__value = T2((T2) __value - (T2) obj.__value + q) % q;
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
	constexpr kAry_Type_T &operator=(int32_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T &operator=(int64_t const obj) noexcept {
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
	constexpr inline bool operator!=(kAry_Type_T const &obj) const noexcept {
		return __value != obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator==(kAry_Type_T const &obj) const noexcept {
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
	constexpr inline bool operator>(kAry_Type_T const &obj) const noexcept {
		return __value > obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator>=(kAry_Type_T const &obj) const noexcept {
		return __value >= obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator<(kAry_Type_T const &obj) const noexcept {
		return __value < obj.__value;
	}

	///
	/// \param obj
	/// \return
	constexpr inline bool operator<=(kAry_Type_T const &obj) const noexcept {
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
								   const uint32_t upper=bits()) noexcept {
		return o1.is_equal(o2, lower, upper);
	}

	///
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_equal(kAry_Type_T const &o,
								   const uint32_t lower=0,
								   const uint32_t upper=bits()) const noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);

		const T mask = compute_mask(lower, upper);
		return (__value & mask) == (o.value() & mask);
	}

	/// only comapres bits
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_greater(kAry_Type_T const &o,
								     const uint32_t lower=0,
								     const uint32_t upper=bits()) const noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);

		const T mask = compute_mask(lower, upper);
		return (__value & mask) > (o.value() & mask);
	}

	/// only compares bits
	/// \param o
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_lower(kAry_Type_T const &o,
	                               const uint32_t lower=0,
	                               const uint32_t upper=bits()) const noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);

		const T mask = compute_mask(lower, upper);
		return (__value & mask) < (o.value() & mask);
	}

	/// only checks if bits are zero
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline bool is_zero(const uint32_t lower=0,
	                              const uint32_t upper=bits()) const noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		return (__value & mask) == T(0);
	}

	/// not really useful, if lower != 0 and upper != bits
	/// \param out
	/// \param in1
	/// \param in2
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline static void add(kAry_Type_T &out,
									 const kAry_Type_T &in1,
									 const kAry_Type_T &in2,
									 const uint32_t lower=0,
									 const uint32_t upper=bits()) noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		const T tmp1 = (in1.value() + in2.value()) & mask;
		const T tmp2 = (out.value() & ~mask) ^ tmp1;
		out.set(tmp2, 0);
	}

	/// not really useful, if lower != 0 and upper != bits
	/// NOTE: ignores carries over the limits.
	constexpr inline static void sub(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
	                                 const kAry_Type_T &in2,
	        						 const uint32_t lower=0,
	                                 const uint32_t upper=bits()) noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		// NOTE: ignores carry here
		const T tmp1 = (in1.value() - in2.value()) & mask;
		const T tmp2 = (out.value() & ~mask) ^ tmp1;
		out.set(tmp2, 0);
	}

	///
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline void neg(const uint32_t lower=0,
	                       	 const uint32_t upper=bits()) const noexcept {
		ASSERT(sizeof(T)*8 > lower);
		ASSERT(sizeof(T)*8 >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		__value ^ mask;
	}

	/// right rotate
	constexpr inline static void rol(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
									 const uint32_t i) noexcept {
		out.__value = rol_T(in1.__value, i);
	}

	/// left rotate
	constexpr inline static void ror(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
									 const uint32_t i) noexcept {
		out.__value = rol_T(in1.__value, i);
	}

	/// left shift
	constexpr inline static void sll(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
									 const uint32_t i) noexcept {
		out.__value = in1.__value << i;
	}

	/// right shift
	constexpr inline static void slr(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
									 const uint32_t i) noexcept {
		out.__value = in1.__value >> i;
	}

	static constexpr inline LimbType add_T(const LimbType a, 
			const LimbType b) noexcept {
		return (a + b) % q;
	}

	static constexpr inline LimbType sub_T(const LimbType a,
			const LimbType b) noexcept {
		return (a + q - b) % q;
	}

	static constexpr inline LimbType mul_T(const LimbType a,
			const LimbType b) noexcept {
		return (T2(a) * T2(b)) % T2(q);
	}

	static constexpr inline LimbType scalar_T(const LimbType a,
			const LimbType b) noexcept {
		return (T2(a) * T2(b)) % T2(q);
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

	// left rotate
	static constexpr inline LimbType rol1_T(const LimbType a) noexcept {
		const T hbit = (a ^ (T(1u) << n)) >> n;
		T ret = a << 1;
		ret ^= hbit;
		return ret;
	}
	
	// right rotate
	static constexpr inline LimbType ror1_T(const LimbType a) noexcept {
		const T bit = (a ^ 1) << n;
		return (a >> 1u) ^ bit;
	}

	// left rotate
	static constexpr inline LimbType rol_T(const LimbType a,
			const uint32_t i) noexcept {
		ASSERT(i < n);
		for (uint32_t j = 0; j < i; j++) {
			rol1_T(a);
		}
	}
	
	// right rotate
	static constexpr inline LimbType ror1_T(const LimbType a,
			const uint32_t i) noexcept {
		ASSERT(i < n);
		for (uint32_t j = 0; j < i; j++) {
			ror1_T(a);
		}
	}

	/// TODO
	/// \param a
	/// \param b
	/// \return
	static constexpr inline uint8x32_t add256_T(const uint8x32_t a, 
			const uint8x32_t b) {
		uint8x32_t ret;
		(void)a;
		(void)b;
		return ret;
	}

	/// \param a
	/// \param b
	/// \return
	static constexpr inline uint8x32_t sub256_T(const uint8x32_t a, 
			const uint8x32_t b) {
		uint8x32_t ret;
		(void)a;
		(void)b;
		return ret;
	}

	///
	/// \param a
	/// \return
	static constexpr inline uint8x32_t mod256_T(const uint8x32_t a) {
		uint8x32_t ret;
		(void)a;
		return ret;
	}

	///
	/// \param a
	/// \param b
	/// \return
	static constexpr inline uint8x32_t mul256_T(const uint8x32_t a, 
			const uint8x32_t b) {
		uint8x32_t ret;
		(void)a;
		(void)b;
		return ret;
	}

	///
	/// \param a
	/// \return
	static constexpr inline uint8x32_t neg256_T(const uint8x32_t a) {
		uint8x32_t ret;
		(void)a;
		return ret;
	}

	T get_value() noexcept { return __value; }
	T value() noexcept { return __value; }

	const T get_value() const noexcept { return __value; }
	const T value() const noexcept { return __value; }
	const T data() const noexcept { return __value; }

	/// not really meaningfully, but needed by the API:
	/// \param i
	/// \return returns the value
	T operator[](const size_t i) noexcept {
		ASSERT(i < bits());
		return value;
	}

	/// also not really meaningfully
	/// \param i
	/// \return
	constexpr inline T get(const size_t i) noexcept {
		ASSERT(i < bits());
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
		ASSERT(i < bits());
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
		ASSERT(i < bits());
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

	///
	static constexpr inline size_t bytes() noexcept {
		return sizeof(T);
	}

	/// NOTE: returns zero id one of the inputs is zero
	/// \param a
	/// \param b
	/// \return
	constexpr static kAry_Type_T gcd(const kAry_Type_T a,
	                                 const kAry_Type_T b) noexcept {
		if (b.is_zero()) { return b; }
		if (a.is_zero()) { return a; }

		// Base case
		if (a == b)
			return a;

		// a is greater
		if (a > b) {
			return gcd(a - b, b);
		}

		return gcd(a, b - a);
	}

	///
	constexpr static kAry_Type_T eea(kAry_Type_T &x, kAry_Type_T &y,
	                       const kAry_Type_T a, const kAry_Type_T b) noexcept {
		// Base Case
		if (a == 0) {
			*x = 0;
			*y = 1;
			return b;
		}

		kAry_Type_T x1, y1; // To store results of recursive call
		kAry_Type_T gcd = gcd(&x1, &y1, b % a, a);

		// Update x and y using results of recursive
		// call
		x = y1 - (b / a) * x1;
		y = x1;

		return gcd;
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
