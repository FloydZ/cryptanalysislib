#ifndef CRYPTANALYSISLIB_HASH_SIMPLE_H
#define CRYPTANALYSISLIB_HASH_SIMPLE_H

#ifndef CRYPTANALYSISLIB_HASH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/hash/hash.h>`"
#endif

#include <cstdint>
#include <type_traits>

#include "math/math.h"
#include "algorithm/bits/popcount.h"
#include "simd/simd.h"

// TODO add docs
// TODO add namespace
// TODO add [[nodiscard]]

/// main comparison class for hash function used within the data containers
/// these hash functions are either optimized for
///	- the case the data is packed together: BinaryVector
/// - the case the data is not packed together like `FqNonPackedVector`
//		and one needs to add values together
template<typename T, const uint32_t ...Ks>
class Hash {};


#ifdef __cpp_static_call_operator
#define CRYPTANALYSISLIB_HASH_STATIC static
#define CRYPTANALYSISLIB_HASH_CONST 
#else
#define CRYPTANALYSISLIB_HASH_STATIC
#define CRYPTANALYSISLIB_HASH_CONST const
#endif

///
/// \tparam T
/// \tparam l lower element (NOT bit)
/// \tparam h upper element (NOT bit)
/// \tparam q modulus
template<std::integral T, // examples:
         const uint32_t l,// = 0
		 const uint32_t h,// = 8u * sizeof(T),
		 const uint32_t q>// = 2u>
class Hash<T, l, h, q>{
private:
	static_assert(q >= 2);
	using H = Hash<T, l, h, q>;
	using R = size_t;

	constexpr static uint32_t qbits = std::max((uint32_t) ceil_log2(q), (uint32_t)1ull);
	constexpr static uint32_t bits = sizeof(T) * 8u;
	static_assert(qbits >= 1);
	static_assert(bits >= 8);

	///
	/// \tparam lprime NOTE: must be the lower bit positions within the limb
	/// \tparam hprime NOTE: must be the upper bit posiition within the limb
	/// \param a
	/// \return
	template<const uint32_t lprime=l*qbits,
             const uint32_t hprime=h*qbits>
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

		/// not so trivial case: the digits in the bits [lprime, hprime)
		/// (`qbits` each) are interpreted as a number in base q,
		/// i.e. digit `i` gets the weight q**i.
		/// NOTE: before, `hprime/qbits` (a digit count) was used as a bit
		/// 	position, `lprime` was ignored and only half of the digits
		/// 	were read, so for q=3 the wrong digits were hashed.
		constexpr uint32_t digits = (hprime - lprime) / qbits;
		constexpr T mask_q = (T(1ull) << qbits) - 1ull;

		T tmp = a >> lprime;
		R ret = 0, ctr = 1;

		#pragma unroll
		for (uint32_t i = 0u; i < digits; ++i) {
			ret += ctr * (tmp & mask_q);
			tmp >>= qbits;
			ctr *= q;
		}

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
		static_assert((hq - lq) <= bits);

		constexpr T lmask = T(-1ull) << lprime;
		constexpr T hmask = T(-1ull) >> ((bits - hprime) % bits);

		// not so easy case: lower limit and upper limit are
		// on seperate limbs
		T data = (a[llimb] & lmask) >> lprime;
		data ^= (a[hlimb] & hmask) << ((bits - lprime) % bits);

		// NOTE: the bits [lq, hq) are now in [0, hq - lq). Before, they were
		// 	returned as is, i.e. not in base q like in the single limb case.
		return compute<0, hq - lq>(data);
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

	CRYPTANALYSISLIB_HASH_STATIC constexpr inline R operator()(const T d) CRYPTANALYSISLIB_HASH_CONST noexcept {
		return compute(d);
	}

    /// \param d[in]:
	CRYPTANALYSISLIB_HASH_STATIC constexpr inline R operator()(const T *d) CRYPTANALYSISLIB_HASH_CONST noexcept {
		return compute(d);
	}

	[[nodiscard]] constexpr static inline R hash(const T d) noexcept {
		return compute(d);
	}

	[[nodiscard]] constexpr static inline R hash(const T *d) noexcept {
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

	// if true: the hash function assumes that the input data
	// is "compressed" together, e.g. there are no zero bits
	// in between two consecutive numbers
	constexpr static bool compressed = true;

	///
	constexpr static uint32_t qbits = std::max((uint64_t) ceil_log2(q), (uint64_t)1ull);
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
		assert(lprime < hprime);
		assert((hprime - lprime) <= (sizeof(T) * 8u));

		/// trivial case: q is a power of two
		if ((cryptanalysislib::popcount::popcount(q) == 1) || compressed) {
			const T diff1 = hprime - lprime;
			assert(diff1 <= bits);
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
		assert(l < h);
		assert(((h- l)*qbits) <= (sizeof(T) * 8u));

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

		assert(llimb <= hlimb);
		assert((hlimb - llimb) <= 1u); // note could be extended

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

    /// \param d[in]:
    /// \param l[in]: inclusive
    /// \param h[in]: exclusive
	CRYPTANALYSISLIB_HASH_STATIC constexpr inline R operator()(const T d,
	                                                           const uint32_t l,
	                                                           const uint32_t h) CRYPTANALYSISLIB_HASH_CONST noexcept {
		return compute(d, l, h);
	}

    /// \param d[in]:
    /// \param l[in]: inclusive
    /// \param h[in]: exclusive
	CRYPTANALYSISLIB_HASH_STATIC constexpr inline R operator()(const T *d,
	                                                           const uint32_t l,
	                                                           const uint32_t h) CRYPTANALYSISLIB_HASH_CONST noexcept {
		return compute(d, l, h);
	}

    /// \param d[in]:
    /// \param l[in]: inclusive
    /// \param h[in]: exclusive
	constexpr static inline R hash(const T d,
	                               const uint32_t l,
	                               const uint32_t h) noexcept {
		return compute(d, l, h);
	}

    /// \param d[in]:
    /// \param l[in]: inclusive
    /// \param h[in]: exclusive
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
		static_assert(l < h);
		static_assert(h <= 128u);
		// NOTE: was `1u << h`, which is a 32 bit shift (UB for h >= 32)
		constexpr __uint128_t mask1 = ~((__uint128_t(1) << l) - 1u);
		constexpr __uint128_t mask2 = h == 128u ? __uint128_t(-1) : ((__uint128_t(1) << h) - 1u);
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
constexpr static inline T extract(const T *v) noexcept {
	constexpr uint32_t BITSIZE = sizeof(T) * 8u;
	static_assert(k_lower < k_higher);
	static_assert(BITSIZE <= 64u);
	// NOTE: was `<= 128`, but the result is a `T`, wider ranges were truncated
	static_assert(k_higher - k_lower <= BITSIZE);
	constexpr uint32_t width = k_higher - k_lower;
	constexpr uint32_t llimb = k_lower / BITSIZE;
	constexpr uint32_t hlimb = (k_higher - 1) / BITSIZE;
	constexpr uint32_t l = k_lower % BITSIZE;
	constexpr __uint128_t mask = (__uint128_t(1) << width) - 1u;

	// NOTE: before, the bits above `k_higher` of the upper limb were not
	// 	masked out, and the 3 limb case shifted by `l` twice
	__uint128_t data = v[llimb];
	if constexpr (llimb != hlimb) {
		data ^= __uint128_t(v[hlimb]) << BITSIZE;
	}

	const T ret = T((data >> l) & mask);
	if constexpr (flip == 0) {
		return ret;
	} else {
		static_assert(k_lower < flip);
		static_assert(flip < k_higher);

		constexpr uint32_t fshift1 = flip - k_lower;
		constexpr uint32_t fshift2 = k_higher - flip;

		// is moment:
		// k_lower          flip                        k_higher
		// [a                b|c                             d]
		// after this transformation:
		// k_higher                     flip            k+lower
		// [c                            d|a                 b]
		// NOTE: before, the flip was ignored in the single limb case, and
		// 	applied to the unshifted and unmasked bits otherwise
		constexpr T fmask = T((T(1ull) << fshift1) - 1ull);// low part

		// move: high -> low ,low -> high
		return T((ret >> fshift1) ^ (T(ret & fmask) << fshift2));
	}
}

#endif
