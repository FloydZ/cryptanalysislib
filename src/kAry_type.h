#ifndef SMALLSECRETLWE_KARY_TYPE_H
#define SMALLSECRETLWE_KARY_TYPE_H

#include <cmath>
#include <cstdint>
#include <iostream>

#include "random.h"

#if __cplusplus > 201709L
#include <concepts>

///
/// \tparam T	Datacontainer to hold atleast one value%q
/// \tparam T2  Data container of the size at least 2*T. Needed to store the multiplication<T2>
template<typename T, typename T2>
concept kAry_TypeAble =
        std::is_integral<T>::value &&
        std::is_integral<T2>::value &&
        requires(T t, T2 t2, uint64_t a) {
	        t + t;
	        t *t;
	        t % t;
	        t % a;
	        sizeof(T) < sizeof(T2);
        };
#endif

/// the IMPORTANT promise of this class is that to all time the the value hold by class field 'value' ___MUST__ be < q
/// Idea: template<typename T=uint16_t, typename T2=uint32_t, const T q = G_q>
/// \tparam T  container type to hold log2(q) bits
/// \tparam T2 container type to hold log2(2q) bits
/// \tparam q  modulus
template<typename T, typename T2, const T q>
#if __cplusplus > 201709L
    requires kAry_TypeAble<T, T2>
#endif
class kAry_Type_T {
public:
	kAry_Type_T() noexcept {
		static_assert(q > 1);
		__value = 0;
	};

	kAry_Type_T(const uint16_t i) noexcept {
		static_assert(q > 2);
		__value = i % q;
	};

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
	void addmul(kAry_Type_T<T, T2, q> const &a, kAry_Type_T<T, T2, q> const &b) noexcept {
		__value = T2(__value + T2(T2(a.__value) * T2(b.__value)) % q) % q;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator&(kAry_Type_T<T, T2, q> obj1, long const obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = (T2(obj1.__value) & T2(obj2)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator^(kAry_Type_T<T, T2, q> obj1, long const obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = (T2(obj1.__value) ^ T2(obj2)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator*(kAry_Type_T<T, T2, q> obj1, long const obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = T((T2(T2(obj1.__value) * T2(obj2))) % q);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator&(const kAry_Type_T<T, T2, q> obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = (T2(obj1.__value) & T2(obj2.__value)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator^(kAry_Type_T<T, T2, q> obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = (T2(obj1.__value) ^ T2(obj2.__value)) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator*(kAry_Type_T<T, T2, q> obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = (T2(T2(obj1.__value) * T2(obj2.__value))) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T<T, T2, q> operator+(kAry_Type_T<T, T2, q> obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = (T2(T2(obj1.__value) + T2(obj2.__value))) % q;
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend kAry_Type_T<T, T2, q> operator-(kAry_Type_T<T, T2, q> obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		kAry_Type_T<T, T2, q> r;
		r.__value = obj1.__value >= obj2.__value ? T2(obj1.__value) - T2(obj2.__value) : obj1.__value + (q - obj2.__value);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend bool operator==(uint64_t obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		return obj1 == obj2.__value;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	friend bool operator!=(uint64_t obj1, kAry_Type_T<T, T2, q> const &obj2) noexcept {
		return obj1 != obj2.__value;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator+=(unsigned int obj) noexcept {
		__value = T2((T2) __value + (T2) obj) % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator+=(kAry_Type_T<T, T2, q> const &obj) noexcept {
		__value = T2((T2) __value + (T2) obj.__value) % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator%=(const uint64_t &obj) noexcept {
		__value %= obj;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T<T, T2, q> &operator%(const uint64_t &obj) noexcept {
		__value %= obj;
		return *this;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator=(kAry_Type_T<T, T2, q> const &obj) noexcept {
		if (this != &obj) {
			__value = obj.__value;
		}
		return *this;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator=(uint32_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator=(uint64_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	kAry_Type_T<T, T2, q> &operator=(int32_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr kAry_Type_T<T, T2, q> &operator=(int64_t const obj) noexcept {
		__value = obj % q;
		return *this;
	}

	//	kAry_Type_T<T, T2, q> &operator=(T const obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	//	kAry_Type_T<T, T2, q> &operator=(unsigned int const obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	//	kAry_Type_T<T, T2, q> &operator=(unsigned long obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	///
	/// \param obj
	/// \return
	inline bool operator!=(kAry_Type_T<T, T2, q> const &obj) const noexcept {
		return __value != obj.__value;
	}

	///
	/// \param obj
	/// \return
	inline bool operator==(kAry_Type_T<T, T2, q> const &obj) const noexcept {
		return __value == obj.__value;
	}

	///
	/// \param obj
	/// \return
	inline bool operator==(uint64_t const &obj) const noexcept {
		return __value == obj;
	}

	///
	/// \param obj
	/// \return
	inline bool operator>(kAry_Type_T<T, T2, q> const &obj) const noexcept {
		return __value > obj.__value;
	}

	///
	/// \param obj
	/// \return
	inline bool operator>=(kAry_Type_T<T, T2, q> const &obj) const noexcept {
		return __value >= obj.__value;
	}

	///
	/// \param obj
	/// \return
	inline bool operator<(kAry_Type_T<T, T2, q> const &obj) const noexcept {
		return __value < obj.__value;
	}

	///
	/// \param obj
	/// \return
	inline bool operator<=(kAry_Type_T<T, T2, q> const &obj) const noexcept {
		return __value <= obj.__value;
	}

	T get_value() noexcept { return __value; }
	T value() noexcept { return __value; }

	const T get_value() const noexcept { return __value; }
	const T value() const noexcept { return __value; }
	const T data() const noexcept { return __value; }
	constexpr uint32_t bits() const noexcept { return constexpr_bits_log2(q); }
	constexpr static T modulo() noexcept { return q; }

private:
	T __value;
};

template<typename T, typename T2, const T q>
std::ostream &operator<<(std::ostream &out, const kAry_Type_T<T, T2, q> &obj) {
	out << signed(obj.get_value());
	return out;
}

//generic abs function for the kAryType
template<class T>
T abs(T in) {
	if constexpr (std::is_integral<T>::value == true) {
		return __builtin_llabs(in);
	} else {
		// TODO apple is crying that double is not an integral type...
		return 0;
		// return T(__builtin_llabs(in.data()));
	}
}

#include "helper.h"
#endif//SMALLSECRETLWE_KARY_TYPE_H
