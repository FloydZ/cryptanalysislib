#ifndef CRYPTANALYSISLIB_SIMPLE_H
#define CRYPTANALYSISLIB_SIMPLE_H

#include <cstdint>
#include <stdlib.h>
#include <type_traits>

#include "math/log.h"
#include "popcount/popcount.h"

template<typename T = uint64_t,
         const uint32_t l = 0,
         const uint32_t h = 8u * sizeof(T),
         const uint32_t q = 2u>
    requires std::is_integral<T>::value
class Hash {
	static_assert(q >= 2);
	static_assert(l < h);
	static_assert(h <= (sizeof(T) * 8u));

public:
	constexpr inline size_t operator()(const T &a) const noexcept {
		if constexpr (q == 2) {
			constexpr T mask1 = T(~((1ul << l) - 1ul));
			constexpr T mask2 = T((1ul << h) - 1ul);
			constexpr T mask = mask1 & mask2;
			return (a & mask) >> l;
		} else {
			constexpr uint32_t qbits = bits_log2(q);

			/// This is the the hash functions passed to the hashmap.
			/// E.g. Whatever this functions returns its used to compute the
			/// bucket of the given element.
			/// \param a output of `Compress`
			/// \return bucket index in the hashmap (so only l positions are hashed)

			/// trivial case: q is a power of two
			if constexpr (cryptanalysislib::popcount::popcount(q) == 1) {
				constexpr T mask = (1ull << (l * qbits)) - 1ull;
				return a & mask;
			}

			/// not so trivial case
			constexpr uint32_t lower = 0, upper = l * qbits;
			constexpr T mask = (~((T(1ul) << lower) - 1ul)) & ((T(1ul) << upper) - 1ul);
			constexpr T mask_q = (1ull << qbits) - 1ull;
			constexpr uint32_t loops = l >> 1u;

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
	}
};

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
#endif//CRYPTANALYSISLIB_SIMPLE_H
