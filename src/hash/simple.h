#ifndef CRYPTANALYSISLIB_HASH_SIMPLE_H
#define CRYPTANALYSISLIB_HASH_SIMPLE_H

#ifndef CRYPTANALYSISLIB_HASH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/hash/hash.h>`"
#endif

#include <cstdint>
#include <type_traits>

#include "math/math.h"
#include "popcount/popcount.h"
#include "simd/simd.h"

/// TODO comments
/// main comparison class for hash function used within
template<typename T, const uint32_t ...Ks>
class Hash {};

///
/// \tparam T
/// \tparam l lower element (NOT bit)
/// \tparam h upper element (NOT bit)
/// \tparam q modulus
template<std::integral T,
         const uint32_t l,// = 0
		 const uint32_t h,// = 8u * sizeof(T),
		 const uint32_t q>// = 2u>
class Hash<T, l, h, q>{
private:
	static_assert(q >= 2);
	using H = Hash<T, l, h, q>;
	using R = size_t;

	constexpr static uint32_t qbits = std::max(bits_log2(q), 1ul);
	constexpr static uint32_t bits = sizeof(T) * 8u;
	static_assert(qbits >= 1);
	static_assert(bits >= 8);

	///
	/// \tparam lprime NOTE: must be the lower bit positions within the limb
	/// \tparam hprime NOTE: must be the upper bit posiition within the limb
	/// \param a
	/// \return
	template<const uint32_t lprime=l*qbits, const uint32_t hprime=h*qbits>
	static constexpr inline R compute(const T &a) noexcept {
		// NOTE: these checks are not valid globally for the whole class
		static_assert(lprime < hprime);
		static_assert((hprime - lprime) <= (sizeof(T) * 8u));

		/// trivial case: q is a power of two
		if constexpr (cryptanalysislib::popcount::popcount(q) == 1u) {
			constexpr T diff1 = hprime - lprime;
			static_assert (diff1 <= bits);
			constexpr T diff2 = bits - diff1;
			constexpr T mask = ((T)-1ull) >> diff2;
			const T b = a >> lprime;
			const T c = b & mask;
			return c;
		}

		/// not so trivial case
		constexpr uint32_t lower = 0, upper = lprime/qbits;
		constexpr T mask = (~((T(1ull) << lower) - 1ull)) & ((T(1ull) << upper) - 1ull);
		constexpr T mask_q = (1ull << qbits) - 1ull;
		constexpr uint32_t loops = (lprime/qbits) >> 1ull;

		uint64_t ctr = q;
		T tmp = (a & mask) >> lower;
		T ret = tmp & mask_q;

		#pragma unroll
		for (uint32_t i = 1u; i < loops; ++i) {
			tmp >>= qbits;
			ret += ctr * (tmp & mask_q);
			ctr *= q;
		}

		// NOTE autocast
		return ret;
	}

	/// \param a
	/// \return
	static constexpr inline R compute(const T *a) noexcept {
		static_assert(l < h);

		constexpr uint32_t lq = l*qbits;
		constexpr uint32_t hq = h*qbits;
		constexpr uint32_t llimb = lq / bits;
		constexpr uint32_t hlimb  = hq%bits == 0u ? llimb : hq / bits;
		constexpr uint32_t lprime = lq % bits;
		constexpr uint32_t hprime = (hq%bits) == 0 ? bits : hq % bits;

		// easy case: lower limit and upper limit
		// are in the same limb
		if constexpr (llimb == hlimb) {
			return compute<lprime, hprime>(a[llimb]);
		}

		static_assert(llimb <= hlimb);
		static_assert((hlimb - llimb) <= 1u); // note could be extended

		constexpr T lmask = T(-1ull) << lprime;
		constexpr T hmask = T(-1ull) >> ((bits - hprime) % bits);

		// not so easy case: lower limit and upper limit are
		// on seperate limbs
		T data = (a[llimb] & lmask) >> lprime;
		data ^= (a[hlimb] & hmask) << ((bits - lprime) % bits);
		return data;
	}

	R __data;

public:
	// the standard constructor cannot be disabled. As this class,
	// should also be usable as a static operator
	constexpr Hash() noexcept : __data(0) {};

	constexpr explicit Hash(const T d) noexcept : __data(compute(d)) {};
	constexpr inline R operator()() const noexcept {
		return __data;
	}
	constexpr inline R operator()(const T d) const noexcept {
		return compute(d);
	}

	constexpr inline R operator()(const T *d) const noexcept {
		return compute(d);
	}
	constexpr static inline R hash(const T d) noexcept {
		return compute(d);
	}

	constexpr static inline R hash(const T *d) noexcept {
		return compute(d);
	}
};

///
/// \tparam T
/// \tparam l
/// \tparam h
/// \tparam q
/// \param a
/// \param b
/// \return
template<std::integral T,
		const uint32_t l,
		const uint32_t h,
		const uint32_t q>
constexpr inline bool operator==(const Hash<T, l, h, q> &a,
								 const Hash<T, l, h, q> &b) noexcept {
	return a.__data == b.__data;
}

///
/// \tparam T
/// \tparam l
/// \tparam h
/// \tparam q
/// \param a
/// \param b
/// \return
template<std::integral T,
		const uint32_t l,
		const uint32_t h,
		const uint32_t q>
constexpr inline bool operator<=(const Hash<T, l, h, q> &a,
								 const Hash<T, l, h, q> &b) noexcept {
	return a.__data <= b.__data;
}


/// \tparam T
/// \tparam q
template<std::integral T,
         const uint32_t q>// = 2u>
class Hash<T, q>{
private:
	static_assert(q >= 2);
	using H = Hash<T, q>;
	using R = size_t;

	// TODO explain
	constexpr static bool compressed = true;

	constexpr static uint32_t qbits = std::max(bits_log2(q), 1ul);
	constexpr static uint32_t bits = sizeof(T) * 8u;
	static_assert(qbits >= 1);
	static_assert(bits >= 8);

	/// \param a
	/// \tparam lprime NOTE: must be the lower bit positions within the limb
	/// \tparam hprime NOTE: must be the upper bit posiition within the limb
	/// \return
	static constexpr inline R compute(const T &a,
	                                  const uint32_t lprime,
	                                  const uint32_t hprime) noexcept {
		ASSERT(lprime < hprime);
		ASSERT((hprime - lprime) <= (sizeof(T) * 8u));

		/// trivial case: q is a power of two
		if ((cryptanalysislib::popcount::popcount(q) == 1) || compressed) {
			const T diff1 = hprime - lprime;
			ASSERT(diff1 <= bits);
			const T diff2 = bits - diff1;
			const T mask = -1ull >> diff2;
			const T b = a >> lprime;
			const T c = b & mask;
			return c;
		}

		/// not so trivial case
		const uint32_t lower = 0, upper = lprime;
		const T mask = (~((T(1ul) << lower) - 1ul)) & ((T(1ul) << upper) - 1ul);
		const T mask_q = (1ull << qbits) - 1ull;
		const uint32_t loops = lprime >> 1u;

		uint64_t ctr = q;
		T tmp = (a & mask) >> lower;
		T ret = tmp & mask_q;

		#pragma unroll
		for (uint32_t i = 1u; i < loops; ++i) {
			tmp >>= qbits;
			ret += ctr * (tmp & mask_q);
			ctr *= q;
		}

		// NOTE autocast
		return ret;
	}

	/// \param a
	/// \return
	static constexpr inline R compute(const T *a,
	                                  const uint32_t l,
	                                  const uint32_t h) noexcept {
		ASSERT(l < h);
		ASSERT(((h- l)*qbits) <= (sizeof(T) * 8u));

		const uint32_t lq = l*qbits;
		const uint32_t hq = h*qbits;
		const uint32_t llimb  = lq / bits;
		const uint32_t hlimb  = hq%bits == 0u ? llimb : hq / bits;
		const uint32_t lprime = lq % bits;
		const uint32_t hprime = (hq%bits) == 0u ? bits : hq % bits;

		// easy case: lower limit and upper limit
		// are in the same limb
		if (llimb == hlimb) {
			return compute(a[llimb], lprime, hprime);
		}

		ASSERT(llimb <= hlimb);
		ASSERT((hlimb - llimb) <= 1u); // note could be extended

		const T lmask = T(-1ull) << lprime;
		const T hmask = T(-1ull) >> ((bits - hprime) % bits);

		// not so easy case: lower limit and upper limit are
		// on seperate limbs
		T data = (a[llimb] & lmask) >> lprime;
		data ^= (a[hlimb] & hmask) << ((bits - lprime) % bits);
		return data;
	}

	R __data;

public:

	constexpr Hash() noexcept : __data(0) {};
	constexpr explicit Hash(const uint64_t d,
	                        const uint32_t l,
	                        const uint32_t h) noexcept :
	   __data(compute(d, l, h)) {};

	constexpr inline R operator()() const noexcept {
		return __data;
	}

	constexpr inline R operator()(const T d,
	                              const uint32_t l,
	                              const uint32_t h) const noexcept {
		return compute(d, l, h);
	}

	constexpr inline R operator()(const T *d,
	                              const uint32_t l,
	                              const uint32_t h) const noexcept {
		return compute(d, l, h);
	}

	constexpr static inline R hash(const T d,
	                               const uint32_t l,
	                               const uint32_t h) noexcept {
		return compute(d, l, h);
	}

	constexpr static inline R hash(const T *d,
	                               const uint32_t l,
	                               const uint32_t h) noexcept {
		return compute(d, l, h);
	}

};



///
/// \tparam T
/// \tparam n
template<typename T, const size_t n>
class Hash<std::array<T, n>> {
private:
	// disable standard constructor
	constexpr Hash() noexcept : __data() {};
	using S = Hash<T>;

public:
	T __data;
	constexpr explicit Hash(const T &d) noexcept : __data(d) {};
};







/// special wrapper class which enforces the compare operator
/// to be following the normal msb/lsb order
/// \tparam S SIMD type: `uint32x32_t` or `uint8x32_t` or `TxN_t`
template<SIMDAble T>
class Hash<T> {
private:
	// disable standard constructor
	constexpr Hash() noexcept : __data() {};
	// internal data type
	using S = Hash<T>;

	// return type
	using R = S;

public:
	// mask compare type
	using C = uint64_t;
	using data_type = T;

	constexpr static C mask = T::LIMBS == 64 ? -1ull : (1ull << T::LIMBS) - 1ull;
	const T *__data;
	constexpr Hash(const T *d) noexcept : __data(d) {};
};

template<SIMDAble T>
constexpr inline bool operator==(const Hash<T> &a,
                                 const Hash<T> &b) noexcept {
	const typename Hash<T>::C t = T::cmp(*a.__data, *b.__data);
	return t == Hash<T>::mask;
}

template<SIMDAble T>
constexpr inline bool operator<(const Hash<T> &a,
                                const Hash<T> &b) noexcept {
	const typename Hash<T>::C t1 = T::lt(*a.__data, *b.__data);
	const typename Hash<T>::C t2 = T::gt(*a.__data, *b.__data);
	return t1 > t2;
}

template<SIMDAble T>
constexpr inline bool operator<=(const Hash<T> &a,
								 const Hash<T> &b) noexcept {
	const typename Hash<T>::C t1 = T::lt(*a.__data, *b.__data);
	const typename Hash<T>::C t2 = T::cmp(*a.__data, *b.__data);
	const typename Hash<T>::C t3 = t1 ^ t2;
	return t3 == Hash<T>::mask;
}







/// not really possible rename to Hash and to add a  concept for `ptr`
/// So for now, just call this function if you need to
/// to hash from a special type, like `KAry<..>`
template<typename L, const uint32_t l, const uint32_t h>
class HashD {
public:
	constexpr inline size_t operator()(const L &k) const noexcept {
		constexpr __uint128_t mask1 = ~((1u << l) - 1u);
		constexpr __uint128_t mask2 = (1u << h) - 1u;
		constexpr __uint128_t mask = mask1 & mask2;
		return ((*(__uint128_t *) k.ptr()) & mask) >> l;
	}
};


/// \tparam k_lower		lower coordinate to extract
/// \tparam k_higher 	higher coordinate (nit included) to extract
/// \tparam flip		if == 0 : nothing happens
/// 					k_lower <= flip <= k_higher:
///							exchanges the bits between [k_lower, ..., flip] and [flip, ..., k_upper]
/// \param v1
/// \param v3
/// \return				v on the coordinates between [k_lower] and [k_higher]
template<typename T, uint32_t k_lower, uint32_t k_higher, uint32_t flip = 0>
static inline T extract(const T *v) noexcept {
	static_assert(k_lower < k_higher);
	static_assert(k_higher - k_lower <= 128u);
	constexpr uint32_t BITSIZE = sizeof(T) * 8u;
	constexpr uint32_t llimb = k_lower / BITSIZE;
	constexpr uint32_t hlimb = (k_higher - 1) / BITSIZE;
	constexpr uint32_t l = k_lower % BITSIZE;
	constexpr uint32_t h = k_higher % BITSIZE;

	constexpr T mask1 = ~((T(1ull) << l) - 1ull);
	constexpr T mask2 = h == uint32_t(0) ? T(-1ull) : ((T(1u) << h) - 1ull);

	if constexpr (llimb == hlimb) {
		constexpr T mask = mask1 & mask2;
		return (v[llimb] & mask) >> l;
	} else {
		__uint128_t data;

		if constexpr (llimb == hlimb - 1) {
			// simple case
			T dl = v[llimb] & mask1;
			T dh = v[hlimb];
			data = dl ^ (__uint128_t(dh) << BITSIZE);
		} else {
			// hard case
			data = *(__uint128_t *) (&v[llimb]);
			data >>= l;
			data ^= (__uint128_t(v[hlimb]) << (128 - l));
		}

		if constexpr (flip == 0) {
			return data >> l;
		} else {
			static_assert(k_lower < flip);
			static_assert(flip < k_higher);

			constexpr uint32_t fshift1 = flip - k_lower;
			constexpr uint32_t fshift2 = k_higher - flip;
			constexpr uint32_t f = fshift1;

			// is moment:
			// k_lower          flip                        k_higher
			// [a                b|c                             d]
			// after this transformation:
			// k_higher                     flip            k+lower
			// [c                            d|a                 b]
			constexpr T fmask1 = ~((T(1ull) << f) - 1ull);// high part
			constexpr T fmask2 = (T(1ull) << f) - 1ull;   // low part

			// move: high -> low ,low -> high
			T data2 = data >> fshift1;
			T data3 = (data & fmask2) << fshift2;
			T data4 = data2 ^ data3;

			return data4;
		}
	}
}

#endif
