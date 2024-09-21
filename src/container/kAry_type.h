#ifndef SMALLSECRETLWE_KARY_TYPE_H
#define SMALLSECRETLWE_KARY_TYPE_H

#include <cmath>
#include <cstdint>
#include <iostream>

#include "helper.h"
#include "math/math.h"
#include "metric.h"
#include "popcount/popcount.h"
#include "print/print.h"
#include "random.h"
#include "simd/simd.h"
#include "alloc/alloc.h"

using namespace cryptanalysislib::metric;

struct FqConfig : public AlignmentConfig {
	// see Fq::mirror for explanation
	constexpr static bool mirror = false;

	// see Fq::arith for explanation
	constexpr static bool arith = true;
	
	// see Fq::lower_is_zero for explanation
	constexpr static bool lower_is_zero = true;
} fqConfig;

/// Container to represent a single element mod q
/// \tparam q modulus
/// \tparam Metric: currently unused
/// \tparam FqConfig: currently unused
template<const uint64_t _q,
         class Metric = HammingMetric,
         const FqConfig &config=fqConfig>
class kAry_Type_T {
public:
	// make the length and modulus of the container public available
	constexpr static uint64_t q = _q;
	[[nodiscard]] constexpr static inline uint64_t modulus() noexcept { return _q; }
	constexpr static uint64_t n = 1;

	/// \return the number of bits needed to store a single element mod q
	[[nodiscard]] constexpr static inline uint64_t bits() noexcept { return ceil_log2(q); }

	/// \return the number of subelements within this container. As this 
	/// 		container only contains a single number its 1.
	[[nodiscard]] constexpr static inline uint64_t length() noexcept { return 1; }

	constexpr static uint32_t qbits = ceil_log2(q);

	constexpr static uint64_t M = computeM_u32(_q);
	// max bytes of T for which `fastmod` is defined
	constexpr static uint64_t M_limit = 0;

	using T = TypeTemplate<q>;
	using T2 = TypeTemplate<__uint128_t(q) * __uint128_t(q)>;

#ifdef USE_AVX512F
	using S = TxN_t<T, 64u / sizeof(T)>;
#else
	using S = TxN_t<T, 32u / sizeof(T)>;
#endif

	// this is needed to make sure that we have enough `bits` in reserve to
	// correctly compute the multiplication.
	static_assert(sizeof(T) <= sizeof(T2), "something odd is going on");
	static_assert(bits() <= (8 * sizeof(T)), "something odd is going on");
	static_assert(q > 1, "mod 1 or 0?");

	// if true, all bit operations are flipped,
	// instead of the lowest bits, all operations are
	// performed on the highest bits
	constexpr static bool mirror = config.mirror;

	// if true, all arith operations ar performed
	// as arithmetic operations, regardless of the
	// given bit boundaries.
	constexpr static bool arith = config.arith;

	// if true, all mask operations ignore the given
	// `k_lower` parameters, and set them to zero.
	// Very useful for SubSet Sum calculations
	constexpr static bool lower_is_zero = config.lower_is_zero;

	static_assert(mirror + arith <= 1, "only one of the two params should be set to `true`");

	/// returns the number of elements stored this container
	constexpr static uint32_t internal_limbs = 1;

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
	/// \param lower lower bound in bits
	/// \param upper upper bound in bits
	/// \return mask
	static constexpr inline const T compute_mask(const uint32_t lower,
	                                             const uint32_t upper) noexcept {
		ASSERT(lower < upper);
		ASSERT(upper <= bits());
		if constexpr (mirror && lower_is_zero) {
			const T mask2 = T(-1u) << (bits() - upper);
			return mask2;
		}

		if constexpr (mirror && !lower_is_zero) {
			const T mask1 = T(-1u) >> lower;
			const T mask2 = T(-1u) << (bits() - upper);
			const T mask = mask1 & mask2;
			return mask;
		}

		if constexpr (!mirror && !lower_is_zero) {
			const T mask1 = T(-1ull) << lower;
			const T mask2 = (1ull << upper) - T(1ull);
			const T mask = mask1 & mask2;
			return mask;
		}

		if constexpr (!mirror && lower_is_zero) {
			const T mask2 = (1ull << upper) - T(1ull);
			return mask2;
		}

		ASSERT(false);
		return 0;
	}

public:

	/// empty constructor, will init to zero
	constexpr kAry_Type_T() noexcept {
		__value = 0;
	};

	/// basic construct
	/// \param i a number of type T
	constexpr kAry_Type_T(const T i) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = fastmod_u32(i, M, q);
		} else {
			__value = i % q;
		}
	};

	///
	/// \param in
	constexpr kAry_Type_T(const kAry_Type_T &in) noexcept {
		this->__value = in.__value;
	}

	///
	inline void random() noexcept {
		__value = rng(q);
	}

	///
	/// \param l
	/// \param u
	inline void random(const T l,
	                   const T u) noexcept {
		if constexpr (arith) {
			__value = rng<T>(l, u);
		} else {
			const T mask = (1ull << u) - 1ull;
			__value == (rng() & mask) << l;
		}
	}

	/// \return
	constexpr T abs() const noexcept {
		return std::abs(__value);
	}

	///	sets this = (this * a) + b
	/// \param a
	/// \param b
	constexpr void addmul(const T &a,
	                      const T &b) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = T(fastmod_u32(T2(__value) + (T2(a) * T2(b)), M, q));
		} else {
			__value = T((T2(__value) + (T2(a) * T2(b))) % q);
		}
	}

	///	sets this = (this * a) + b
	/// \param a
	/// \param b
	constexpr void addmul(kAry_Type_T const &a,
	                      kAry_Type_T const &b) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = T(fastmod_u32(T2(__value) + (T2(a.__value) * T2(b.__value)), M, q));
		} else {
			__value = T((T2(__value) + T2(a.__value) * T2(b.__value)) % q);
		}
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return obj1 & obj2
	constexpr inline friend kAry_Type_T operator&(const kAry_Type_T obj1,
	                                              const uint64_t obj2) noexcept {
		kAry_Type_T r;
		r.__value = T((T2(obj1.__value) & T2(obj2)) % q);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator^(const kAry_Type_T obj1,
	                                              const uint64_t obj2) noexcept {
		kAry_Type_T r;
		r.__value = T((T2(obj1.__value) ^ T2(obj2 % q)) % q);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator-(const kAry_Type_T obj1,
	                                              const uint64_t obj2) noexcept {
		kAry_Type_T r;
		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) - T2(fastmod_u32(obj2, M, q)), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) + (T2(q)) - T2(obj2 % q))) % q);
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator+(const kAry_Type_T obj1,
	                                              const uint64_t obj2) noexcept {
		kAry_Type_T r;
		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) + T2(fastmod_u32(obj2, M, q)), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) + T2(obj2 % q))) % q);
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator*(const kAry_Type_T obj1,
	                                              const uint64_t obj2) noexcept {
		kAry_Type_T r;

		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) * T2(fastmod_u32(obj2, M, q)), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) * T2(obj2 % q))) % q);
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator/(const kAry_Type_T obj1,
	                                              const uint64_t obj2) noexcept {
		kAry_Type_T r;

		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) / T2(fastmod_u32(obj2, M, q)), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) / T2(obj2 % q))) % q);
		}
		return r;
	}


	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator&(const kAry_Type_T &obj1,
	                                              const kAry_Type_T &obj2) noexcept {

		kAry_Type_T r;
		r.__value = T((T2(obj1.__value) & T2(obj2.__value)) % q);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator^(const kAry_Type_T &obj1,
	                                              const kAry_Type_T &obj2) noexcept {
		kAry_Type_T r;
		r.__value = T((T2(obj1.__value) ^ T2(obj2.__value)) % q);
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator/(const kAry_Type_T &obj1,
	                                              const kAry_Type_T &obj2) noexcept {
		kAry_Type_T r;

		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) / T2(obj2.__value), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) / T2(obj2.__value))) % q);
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator*(const kAry_Type_T &obj1,
	                                              const kAry_Type_T &obj2) noexcept {
		kAry_Type_T r;
		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) * T2(obj2.__value), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) * T2(obj2.__value))) % q);
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator+(const kAry_Type_T &obj1,
	                                              const kAry_Type_T &obj2) noexcept {
		kAry_Type_T r;
		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) + T2(obj2.__value), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) + T2(obj2.__value))) % q);
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	constexpr inline friend kAry_Type_T operator-(kAry_Type_T obj1,
	                                              kAry_Type_T const &obj2) noexcept {
		kAry_Type_T r;

		if constexpr (sizeof(T) <= M_limit) {
			r.__value = T(fastmod_u32(T2(obj1.__value) + T2(q) - T2(obj2.__value), M, q));
		} else {
			r.__value = T((T2(T2(obj1.__value) + T2(q) - T2(obj2.__value))) % T2(q));
		}
		return r;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	[[nodiscard]] constexpr inline friend bool operator==(const uint64_t obj1,
	                                                      kAry_Type_T const &obj2) noexcept {
		return obj1 == obj2.__value;
	}

	///
	/// \param obj1
	/// \param obj2
	/// \return
	[[nodiscard]] constexpr inline friend bool operator!=(uint64_t obj1, kAry_Type_T const &obj2) noexcept {
		return obj1 != obj2.__value;
	}

	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator+=(const T obj) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = T(fastmod_u32(T2(__value) + T2(fastmod_u32(obj, M, q)), M, q));
		} else {
			__value = T(T2(T2(__value) + T2(obj % q)) % q);
		}
		return *this;
	}

	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator+=(kAry_Type_T const &obj) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = T(fastmod_u32(T2(__value) + T2(fastmod_u32(obj.__value, M, q)), M, q));
		} else {
			__value = T(T2((T2) __value + (T2) obj.__value) % q);
		}
		return *this;
	}

	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator-=(const T obj) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = T(fastmod_u32(T2(__value) + T2(q) - T2(fastmod_u32(obj, M, q)), M, q));
		} else {
			__value = T(T2(T2(__value) + T2(obj % q)) % q);
		}
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator-=(kAry_Type_T const &obj) noexcept {
		if constexpr (sizeof(T) <= M_limit) {
			__value = T(fastmod_u32(T2(__value) + T2(q) - T2(fastmod_u32(obj.__value, M, q)), M, q));
		} else {
			__value = T(T2(T2(__value) + (T2(q) - T2(obj.__value % q))) % q);
		}
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator%=(const uint64_t &obj) noexcept {
		__value %= obj;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator%(const uint64_t &obj) noexcept {
		__value %= obj;
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator=(kAry_Type_T const &obj) noexcept {
		if (this != &obj) {
			__value = obj.__value;
		}

		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator=(uint32_t const obj) noexcept {
		__value = T(obj % q);
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator=(uint64_t const obj) noexcept {
		__value = T(obj % q);
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator=(int32_t const obj) noexcept {
		__value = T(obj % q);
		return *this;
	}

	///
	/// \param obj
	/// \return
	constexpr inline kAry_Type_T &operator=(int64_t const obj) noexcept {
		__value = T(obj % q);
		return *this;
	}

	//	kAry_Type_T &operator=(T const obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	//	kAry_Type_T &operator=(unsigned int const obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	//	kAry_Type_T &operator=(unsigned long obj) {
	//		__value = obj % q;
	//		return *this;
	//	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator!=(kAry_Type_T const &obj) const noexcept {
		return __value != obj.__value;
	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator==(kAry_Type_T const &obj) const noexcept {
		return __value == obj.__value;
	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator==(uint64_t const &obj) const noexcept {
		return __value == obj;
	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator>(kAry_Type_T const &obj) const noexcept {
		return __value > obj.__value;
	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator>=(kAry_Type_T const &obj) const noexcept {
		return __value >= obj.__value;
	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator<(kAry_Type_T const &obj) const noexcept {
		return __value < obj.__value;
	}

	///
	/// \param obj
	/// \return
	[[nodiscard]] constexpr inline bool operator<=(kAry_Type_T const &obj) const noexcept {
		return __value <= obj.__value;
	}

	///
	/// \param o
	/// \param lower inclusive
	/// \param upper exclusive
	/// \return this[lower, upper) == o[lower, upper)
	[[nodiscard]] static constexpr inline bool cmp(kAry_Type_T const &o1,
	                                               kAry_Type_T const &o2,
	                                               const uint32_t lower = 0,
	                                               const uint32_t upper = bits()) noexcept {
		return o1.is_equal(o2, lower, upper);
	}

	/// only compares bits
	/// \param o
	/// \param lower inclusive
	/// \param upper exclusive
	/// \return this[lower, upper) == o[lower, upper)
	[[nodiscard]] constexpr inline bool is_equal(kAry_Type_T const &o,
	                                             const uint32_t lower = 0,
	                                             const uint32_t upper = bits()) const noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		const T mask = compute_mask(lower, upper);
		return (__value & mask) == (o.value() & mask);
	}

	template<const uint32_t lower, const uint32_t upper>
	[[nodiscard]] constexpr inline bool is_equal(kAry_Type_T const &o) const noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		constexpr T mask = compute_mask(lower, upper);
		return (__value & mask) == (o.value() & mask);
	}

	/// only compares bits
	/// \param o
	/// \param lower inclusive
	/// \param upper exclusive
	/// \return this[lower, upper) > o[lower, upper)
	[[nodiscard]] constexpr inline bool is_greater(kAry_Type_T const &o,
	                                               const uint32_t lower = 0,
	                                               const uint32_t upper = bits()) const noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		const T mask = compute_mask(lower, upper);
		return (__value & mask) > (o.value() & mask);
	}

	template<const uint32_t lower, const uint32_t upper>
	[[nodiscard]] constexpr inline bool is_greater(kAry_Type_T const &o) const noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		constexpr T mask = compute_mask(lower, upper);
		return (__value & mask) > (o.value() & mask);
	}

	/// only compares bits
	/// \param o
	/// \param lower inclusive
	/// \param upper exclusive
	/// \return this[lower, upper) < o[lower, upper)
	[[nodiscard]] constexpr inline bool is_lower(kAry_Type_T const &o,
	                                             const uint32_t lower = 0,
	                                             const uint32_t upper = bits()) const noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		const T mask = compute_mask(lower, upper);
		const T t1 = __value & mask;
		const T t2 = o.value() & mask;
		return t1 < t2;
	}

	/// only compares bits
	/// \param o
	/// \param lower inclusive
	/// \param upper exclusive
	/// \return this[lower, upper) < o[lower, upper)
	template<const uint32_t lower, const uint32_t upper>
	[[nodiscard]] constexpr inline bool is_lower(kAry_Type_T const &o) const noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		constexpr T mask = compute_mask(lower, upper);
		const T t1 = __value & mask;
		const T t2 = o.value() & mask;
		return t1 < t2;
	}

	/// only checks if bits are zero
	/// \param lower
	/// \param upper
	/// \return
	[[nodiscard]] constexpr inline bool is_zero(const uint32_t lower = 0,
	                                            const uint32_t upper = bits()) const noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		const T mask = compute_mask(lower, upper);
		return T(__value & mask) == T(0);
	}


	///
	/// \tparam lower
	/// \tparam upper
	/// \param lower
	/// \param upper
	/// \return
	template<const uint32_t lower, const uint32_t upper>
	[[nodiscard]] constexpr inline bool is_zero() const noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		constexpr T mask = compute_mask(lower, upper);
		return T(__value & mask) == T(0);
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
	                                 const uint32_t lower = 0,
	                                 const uint32_t upper = bits()) noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		if constexpr (arith) {
			out.__value = T(((T2(in1.__value) + T2(in2.__value)) % q));
		} else {
			const T mask = compute_mask(lower, upper);
			const T tmp1 = (in1.value() ^ in2.value()) & mask;
			const T tmp2 = (out.value() & ~mask) ^ tmp1;
			out.set(tmp2, 0);
		}
	}

	/// not really useful, if lower != 0 and upper != bits
	/// \param out
	/// \param in1
	/// \param in2
	/// \param lower
	/// \param upper
	/// \return
	template<const uint32_t lower, const uint32_t upper>
	constexpr inline static void add(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
	                                 const kAry_Type_T &in2) noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		if constexpr (arith) {
			out.__value = T(((T2(in1.__value) + T2(in2.__value)) % q));
		} else {
			constexpr T mask = compute_mask(lower, upper);
			const T tmp1 = (in1.value() + in2.value()) & mask;
			const T tmp2 = (out.value() & ~mask) ^ tmp1;
			out.set(tmp2, 0);
		}
	}

	/// not really useful, if lower != 0 and upper != bits
	/// NOTE: ignores carries over the limits.
	constexpr inline static void sub(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
	                                 const kAry_Type_T &in2,
	                                 const uint32_t lower = 0,
	                                 const uint32_t upper = bits()) noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		if constexpr (arith) {
			out.__value = ((T2(in1.__value) + T2(q) - T2(in2.__value)) % q);
		} else {
			// NOTE: ignores carry here
			constexpr T mask = compute_mask(lower, upper);
			const T tmp1 = (in1.value() - in2.value()) & mask;
			const T tmp2 = (out.value() & ~mask) ^ tmp1;
			out.set(tmp2, 0);
		}
	}

	///
	/// \tparam lower
	/// \tparam upper
	/// \param out
	/// \param in1
	/// \param in2
	/// \return
	template<const uint32_t lower, const uint32_t upper>
	constexpr inline static void sub(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
	                                 const kAry_Type_T &in2) noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		if constexpr (arith) {
			out.__value = ((T2(in1.__value) + T2(q) - T2(in2.__value)) % q);
		} else {
			// NOTE: ignores carry here
			constexpr T mask = compute_mask(lower, upper);
			const T tmp1 = (in1.value() - in2.value()) & mask;
			const T tmp2 = (out.value() & ~mask) ^ tmp1;
			out.set(tmp2, 0);
		}
	}

	///
	/// \param out
	/// \param in1
	/// \param in2
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline static void mul(kAry_Type_T &out,
									 const kAry_Type_T &in1,
									 const kAry_Type_T &in2,
									 const uint32_t lower = 0,
									 const uint32_t upper = bits()) noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		if constexpr (arith) {
			out.__value = ((T2(in1.__value) * T2(in2.__value)) % q);
		} else {
			// NOTE: ignores carry here
			const T mask = compute_mask(lower, upper);
			const T tmp1 = (in1.value() ^ in2.value()) & mask;
			const T tmp2 = (out.value() & ~mask) ^ tmp1;
			out.set(tmp2, 0);
		}
	}

	template<const uint32_t lower, const uint32_t upper>
	constexpr inline static void mul(kAry_Type_T &out,
									 const kAry_Type_T &in1,
									 const kAry_Type_T &in2) noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		if constexpr (arith) {
			out.__value = ((T2(in1.__value) * T2(in2.__value)) % q);
		} else {
			// NOTE: ignores carry here
			constexpr T mask = compute_mask(lower, upper);
			const T tmp1 = (in1.value() * in2.value()) & mask;
			const T tmp2 = (out.value() & ~mask) ^ tmp1;
			out.set(tmp2, 0);
		}
	}

	///
	/// \param out
	/// \param in1
	/// \param in2
	/// \param lower
	/// \param upper
	/// \return
	constexpr inline static void scalar(kAry_Type_T &out,
										const kAry_Type_T &in1,
										const DataType &in2,
	                                    const uint32_t lower=0,
	                                    const uint32_t upper=bits()) noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		if constexpr (arith) {
			out.__value = ((T2(in1.__value) * T2(in2 % q)) % q);
		} else {
			const T mask = compute_mask(lower, upper);
			// not implemented
			ASSERT(false);
		}
	}

	///
	/// \tparam lower
	/// \tparam upper
	/// \param out
	/// \param in1
	/// \param in2
	/// \return
	template<const uint32_t lower, const uint32_t upper>
	constexpr inline static void scalar(kAry_Type_T &out,
									 const kAry_Type_T &in1,
									 const DataType &in2) noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		if constexpr (arith) {
			out.__value = ((T2(in1.__value) * T2(in2 % q)) % q);
		} else {
			// not implemented
			constexpr T mask = compute_mask(lower, upper);
			ASSERT(false);
		}
	}

	/// \param lower
	/// \param upper
	/// \return
	constexpr inline void neg(const uint32_t lower = 0,
	                          const uint32_t upper = bits()) noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		if constexpr (arith) {
			__value = ((q - __value) % q);
		} else {
			const T mask = compute_mask(lower, upper);
			__value ^= mask;
			__value &= mask;
		}
	}

	template<const uint32_t lower, const uint32_t upper>
	constexpr inline void neg() noexcept {
		static_assert(sizeof(T) * 8 > lower);
		static_assert(sizeof(T) * 8 >= upper);
		static_assert(lower < upper);
		static_assert(upper <= bits());

		if constexpr (arith) {
			__value = ((q - __value) % q);
		} else {
			constexpr T mask = compute_mask(lower, upper);
			__value ^= mask;
		}
	}

	constexpr inline void popcnt(const uint32_t lower = 0,
	                             const uint32_t upper = bits()) noexcept {
		ASSERT(sizeof(T) * 8 > lower);
		ASSERT(sizeof(T) * 8 >= upper);
		ASSERT(lower < upper);
		ASSERT(upper <= bits());

		if constexpr (arith) {
			__value = ((q - __value) % q);
		} else {
			const T mask = compute_mask(lower, upper);
			__value ^= mask;
		}
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
		out.__value = ror_T(in1.__value, i);
	}

	/// left shift (binary)
	constexpr inline static void sll(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
	                                 const uint32_t i) noexcept {
		out.__value = in1.__value << i;
	}

	/// right shift (binary)
	constexpr inline static void slr(kAry_Type_T &out,
	                                 const kAry_Type_T &in1,
	                                 const uint32_t i) noexcept {
		out.__value = in1.__value >> i;
	}

	[[nodiscard]] static constexpr inline LimbType add_T(const LimbType a,
	                                                     const LimbType b) noexcept {
		return (a + b) % q;
	}

	[[nodiscard]] static constexpr inline LimbType sub_T(const LimbType a,
	                                                     const LimbType b) noexcept {
		return (a + q - b) % q;
	}

	[[nodiscard]] static constexpr inline LimbType mul_T(const LimbType a,
	                                                     const LimbType b) noexcept {
		return (T2(a) * T2(b)) % T2(q);
	}

	[[nodiscard]] static constexpr inline LimbType scalar_T(const LimbType a,
	                                                        const LimbType b) noexcept {
		return (T2(a) * T2(b)) % T2(q);
	}

	[[nodiscard]] static constexpr inline LimbType mod_T(const LimbType a) noexcept {
		return a % q;
	}

	[[nodiscard]] static constexpr inline LimbType neg_T(const LimbType a) noexcept {
		return (q - a) % q;
	}

	[[nodiscard]] static constexpr inline LimbType popcnt_T(const LimbType a) noexcept {
		return cryptanalysislib::popcount::popcount(a);
	}

	// left rotate
	[[nodiscard]] static constexpr inline LimbType rol1_T(const LimbType a) noexcept {
		const T hbit = (a ^ (T(1u) << n)) >> n;
		T ret = a << 1;
		ret ^= hbit;
		return ret;
	}

	// right rotate
	[[nodiscard]] static constexpr inline LimbType ror1_T(const LimbType a) noexcept {
		const T bit = (a ^ 1) << n;
		return (a >> 1u) ^ bit;
	}

	// left rotate
	[[nodiscard]] static constexpr inline LimbType rol_T(const LimbType a,
	                                                     const uint32_t i) noexcept {
		ASSERT(i < n);
		for (uint32_t j = 0; j < i; j++) {
			rol1_T(a);
		}
	}

	// right rotate
	[[nodiscard]] static constexpr inline LimbType ror1_T(const LimbType a,
	                                                      const uint32_t i) noexcept {
		ASSERT(i < n);
		for (uint32_t j = 0; j < i; j++) {
			ror1_T(a);
		}
	}

	/// NOTE: assumes that
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] static constexpr inline S add256_T(const S a,
	                                                 const S b) {
		S ret = S::add(a, b);
		ret = mod256_T(ret);
		return ret;
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] static constexpr inline S sub256_T(const S a,
	                                                 const S b) {
		S ret = S::sub(a, b);
		ret = mod256_T(ret);
		return ret;
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] static constexpr inline S mod256_T(const S a) {
		S ret;
		for (int i = 0; i < S::LIMBS; ++i) {
			ret.d[i] = a.d[i] % q;
		}
		return ret;
	}

	///
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] static constexpr inline S mul256_T(const S a,
	                                                 const S b) {
		S ret = S::mullo(a, b);
		ret = mod256_T(ret);
		return ret;
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] static constexpr inline S neg256_T(const S a) {
		S Q = S::set1(q);
		S ret = S::sub(Q, a);
		ret = mod256_T(ret);
		return ret;
	}

	[[nodiscard]] constexpr inline T get_value() noexcept { return __value; }
	[[nodiscard]] constexpr inline T value() noexcept { return __value; }

	[[nodiscard]] constexpr inline const T get_value() const noexcept { return __value; }
	[[nodiscard]] constexpr inline const T value() const noexcept { return __value; }
	[[nodiscard]] constexpr inline const T data() const noexcept { return __value; }

	/// not really meaningfully, but needed by the API:
	/// \param i
	/// \return returns the value
	[[nodiscard]] constexpr inline T operator[](const size_t i) noexcept {
		return get(i);
	}

	/// not really meaningfully, but needed by the API:
	/// \param i
	/// \return returns the value
	[[nodiscard]] constexpr inline T operator[](const size_t i) const noexcept {
		return get(i);
	}

	/// also not really meaningfully
	/// \param i
	/// \return
	[[nodiscard]] constexpr inline T get(const size_t i) noexcept {
		ASSERT(i < bits());
		(void) i;
		return __value;
	}

	/// also not really meaningfully
	/// \param i
	/// \return
	[[nodiscard]] constexpr inline T get(const size_t i) const noexcept {
		ASSERT(i < bits());
		(void) i;
		return __value;
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
	constexpr inline void zero(const uint32_t l = 0,
	                           const uint32_t h = bits()) noexcept {
		ASSERT(l < h);
		ASSERT(h <= bits());
		const T mask = ~compute_mask(l, h);
		__value &= mask;
	}

	constexpr inline void one(const uint32_t l = 0,
	                          const uint32_t h = bits()) noexcept {
		ASSERT(l < h);
		ASSERT(h <= bits());
		(void) l;
		(void) h;
		__value = 1;
	}
	constexpr inline void minus_one(const uint32_t l = 0,
	                                const uint32_t h = bits()) noexcept {
		ASSERT(l < h);
		ASSERT(h <= bits());
		const T mask = compute_mask(l, h);
		__value |= mask;
	}

	///
	/// \return
	[[nodiscard]] constexpr inline T *ptr() noexcept {
		return &__value;
	}
	[[nodiscard]] constexpr inline const T *ptr() const noexcept {
		return &__value;
	}

	/// NOTE: not really useful: i is ignored
	/// \param i
	/// \return
	[[nodiscard]] constexpr inline T ptr(const size_t i) noexcept {
		ASSERT(i < bits());
		return __value;
	}
	[[nodiscard]] constexpr inline const T ptr(const size_t i) const noexcept {
		ASSERT(i < bits());
		return __value;
	}

	constexpr void print_binary(const uint32_t lower = 0,
	                            const uint32_t upper = bits()) const noexcept {
		ASSERT((8 * sizeof(T)) > lower);
		ASSERT((8 * sizeof(T)) >= upper);
		ASSERT(lower < upper);
		const T mask = compute_mask(lower, upper);
		const T tmp = (__value & mask) >> lower;
		cryptanalysislib::print_binary(tmp);
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

	/// returns the greates common divisor, and x,y s.t. gcd= x*a + y*b;
	constexpr static kAry_Type_T eea(kAry_Type_T &x,
	                                 kAry_Type_T &y,
	                                 const kAry_Type_T a,
	                                 const kAry_Type_T b) noexcept {
		// Base Case
		if (a == 0) {
			*x = 0;
			*y = 1;
			return b;
		}

		kAry_Type_T x1, y1;// To store results of recursive call
		kAry_Type_T gcd = gcd(&x1, &y1, b % a, a);

		// Update x and y using results of recursive
		// call
		x = y1 - (b / a) * x1;
		y = x1;

		return gcd;
	}


	/// NOTE: lower and upper are ignored
	constexpr void print(const uint32_t lower = 0,
	                     const uint32_t upper = length()) const noexcept {
		(void) lower;
		(void) upper;
		std::cout << __value << std::endl;
	}

	[[nodiscard]] static constexpr inline bool binary() noexcept {
		return q == 2;
	}

	/// returns the number of elements stored this container
	[[nodiscard]] static constexpr inline size_t size() noexcept {
		return 1;
	}

	/// returns the number of elements stored this container
	[[nodiscard]] static constexpr inline size_t limbs() noexcept {
		return 1;
	}

	/// returns size of a single element in this container in bits
	[[nodiscard]] static constexpr inline size_t sub_container_size() noexcept {
		return 1;
	}

	///
	[[nodiscard]] static constexpr inline size_t bytes() noexcept {
		return sizeof(T);
	}
	[[nodiscard]] inline static constexpr bool optimized() noexcept { return true; };

	///
	/// \tparam l bit position
	/// \tparam h bit positin
	/// \return
	template<const uint32_t l, const uint32_t h>
	[[nodiscard]] constexpr inline size_t hash() const noexcept {
		static_assert(l < h);
		static_assert(h <= bits());
		constexpr T diff1 = h - l;
		static_assert(diff1 <= bits());
		if constexpr (diff1 == bits()) {
			return __value;
		}

		constexpr T mask = (T(1ull) << diff1) - T(1ull);
		const T b = __value >> l;
		const T c = b & mask;
		return c;
	}

	///
	/// \tparam l bit position
	/// \tparam h bit positin
	/// \return
	[[nodiscard]] constexpr inline size_t hash(const uint32_t l,
	                                           const uint32_t h) const noexcept {
		ASSERT(l < h);
		ASSERT(h <= bits());
		const T diff1 = h - l;
		ASSERT(diff1 <= bits());
		const T mask = diff1 == bits() ? T(-1ull) : (T(1ull) << diff1) - T(1ull);
		const T b = __value >> l;
		const T c = b & mask;
		return c;
	}

	[[nodiscard]] constexpr inline size_t hash() const noexcept {
		return __value;
	}

	template<const uint32_t l, const uint32_t h>
	constexpr static bool is_hashable() noexcept {
		if constexpr (h == l) { return false; }
		constexpr size_t t1 = h - l;
		return t1 <= 64u;
	}
	constexpr static bool is_hashable(const uint32_t l,
	                                  const uint32_t h) noexcept {
		ASSERT(h > l);
		const size_t t1 = h - l;
		return t1 <= 64u;
	}
	/// prints some information about this class
	constexpr static void info() noexcept {
		std::cout << "{ name: \"kAry_Type\""
		          << ", q: " << q
		          << ", n: " << n
		          << ", mirror: " << mirror
		          << ", arith: " << arith
		          << ", sizeof(T): " << sizeof(T)
		          << ", sizeof(T2): " << sizeof(T2)
		          << " }" << std::endl;
	}
private:
	T __value;
};

//generic abs function for the kAryType
template<class T>
T abs(T in) {
	if constexpr (std::is_integral<T>::value == true) {
		return __builtin_llabs(in);
	} else {
		return T(__builtin_llabs(in.data()));
	}
}

template<const uint64_t _q,
         class Metric = HammingMetric>
std::ostream &operator<<(std::ostream &out, const kAry_Type_T<_q, Metric> &obj) {
	constexpr static bool bin = true;
	using S = kAry_Type_T<_q, Metric>;

	if constexpr (bin) {
		uint64_t tmp = obj.value();
		for (size_t i = 0; i < S::bits(); ++i) {
			std::cout << (tmp & 1u);
			tmp >>= 1u;
		}
		out << " (" << std::dec << (uint64_t) obj.value()
		    << ", 0x" << std::hex << (uint64_t) obj.value() << ")"
		    << std::dec;
	} else {
		out << obj.value();
	}
	return out;
}

#endif//SMALLSECRETLWE_KARY_TYPE_H
