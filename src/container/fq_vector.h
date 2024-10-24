#ifndef SMALLSECRETLWE_FQ_VECTOR_H
#define SMALLSECRETLWE_FQ_VECTOR_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "container/common.h"
#include "helper.h"
#include "math/math.h"
#include "algorithm/bits/popcount.h"
#include "random.h"
#include "simd/simd.h"
#include "hash/hash.h"

using namespace cryptanalysislib;

struct FqVectorMetaConfig : public AlignmentConfig {
};
constexpr static FqVectorMetaConfig fqVectorMetaConfig;

/// \tparam T base type
/// \tparam n size of the fq vector space=number of elements
/// \tparam q field size
template<const uint32_t _n,
		 const uint64_t _q,
         typename T=uint64_t,
		 const FqVectorMetaConfig &config=fqVectorMetaConfig>
#if __cplusplus > 201709L
    requires std::is_integral<T>::value
#endif
class FqNonPackedVectorMeta {
public:
	// internal data length. Used in the template system to pass through this information
	constexpr static uint64_t n = _n;
	constexpr static inline uint64_t length() { return n; }
	constexpr static uint16_t internal_limbs = _n;

	constexpr static uint64_t q = _q;
	constexpr static inline uint64_t modulus() { return q; }
	constexpr static uint32_t qbits = ceil_log2(q);

	// Needed for the internal template system.
	typedef T DataType;
	typedef T LimbType;

	typedef FqNonPackedVectorMeta ContainerType;

	//
	constexpr static size_t nr_of_limbs_in_S = limbs<T>();
	using S = TxN_t<T, nr_of_limbs_in_S>;


	/// simple hash function
	/// \tparam l
	/// \tparam h
	/// \return
	template<const uint32_t l, const uint32_t h>
	[[nodiscard]] constexpr inline auto hash() const noexcept {
		static_assert(l < h);
		static_assert(h <= length());
		static_assert(((h-l)*qbits) <= 64);

		__uint128_t d = __data[l];
		uint32_t shift = qbits;
		for (uint32_t i = 1; i < (h - l); i++) {
			d ^= __uint128_t(__data[i + l]) << shift;
			shift += qbits;
		}

		constexpr uint64_t mask = (1ull << ((h-l)*qbits)) - 1ull;
		const uint64_t t1 = d;
		const uint64_t t2 = t1 & mask;
		return t2;
	}

	/// \param l
	/// \param h
	/// \return
	[[nodiscard]] constexpr inline auto hash(const uint32_t l,
	                                         const uint32_t h) const noexcept {
		ASSERT(l < h);
		ASSERT(h <= length());
		ASSERT(((h-l)*qbits) <= 64);

		__uint128_t d = __data[l];
		uint32_t shift = qbits;
		for (uint32_t i = 1; i < (h - l); i++) {
			d ^= __uint128_t(__data[i + l]) << shift;
			shift += qbits;
		}

		const uint64_t mask = (1ull << ((h-l)*qbits)) - 1ull;
		const uint64_t t1 = d;
		const uint64_t t2 = t1 & mask;
		return t2;
	}
	[[nodiscard]] constexpr inline auto hash() const noexcept {
		return *this;
		// using S_ = TxN_t<T, limbs()>;
		// const S_ *s = (S_ *)ptr();
		// const auto t = Hash<S_>(s);
		// return t;
	}

	template<const uint32_t l, const uint32_t h>
	constexpr static bool is_hashable() noexcept {
		static_assert(h > l);
		constexpr size_t t1 = h-l;
		constexpr size_t t2 = t1*qbits;

		return t2 <= 64u;
	}
	constexpr static bool is_hashable(const uint32_t l,
									  const uint32_t h) noexcept {
		ASSERT(h > l);
		const size_t t1 = h-l;
		const size_t t2 = t1*qbits;

		return t2 <= 64u;
	}

	/// zeros our the whole container
	/// \return nothing
	constexpr inline void zero(const uint32_t l=0,
	                           const uint32_t h=length()) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = l; i < h; i++) {
			__data[i] = T(0);
		}
	}

	/// set everything one
	/// \return nothing
	constexpr inline void one(const uint32_t l=0,
	                          const uint32_t h=length()) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = l; i < h; i++) {
			__data[i] = T(1);
		}
	}

	// sets everything
	constexpr inline void minus_one(const uint32_t l=0,
	                                const uint32_t h=length()) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = l; i < h; i++) {
			__data[i] = T(-1ull);
		}
	}

	/// generates rng coordinates
	/// \param k_lower lower coordinate to start from
	/// \param k_higher higher coordinate to stop. Not included.
	void random(const uint32_t k_lower = 0,
	            const uint32_t k_higher = length()) noexcept {
		ASSERT(k_lower < k_higher);
		ASSERT(k_higher <= length());

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_higher; i++) {
			__data[i] = rng(q);
		}
	}


	/// generates a rng weight `w` vector with `w` != 0
	/// \param w we
	/// \param m
	/// \param offset start offset of the first error position
	/// \return
	constexpr void random_with_weight(const uint32_t w,
									  const uint32_t m=length(),
	                                  const uint32_t offset=0) noexcept {
		ASSERT(w <= m);
		ASSERT(m+offset <= length());
		zero();

		// chose first
		for (uint32_t i = 0; i < w; ++i) {
			if constexpr (q == 2) {
				set(1, i + offset);
			} else {
				const auto d = rng<T>(1, q - 1u);
				set(d, i + offset);
			}
		}

		// early exit
		if (w == m) { return; }

		// now permute
		for (uint64_t i = 0; i < m; ++i) {
			uint64_t pos = rng() % (m - i);
			const auto t = get(i+offset);
			set(get(i + pos+offset), i+offset);
			set(t, i + pos+offset);
		}
	}

	/// checks if every dimension is zero
	/// \return true/false
	[[nodiscard]] constexpr bool is_zero(const uint32_t k_lower = 0,
	                                     const uint32_t k_higher = length()) const noexcept {
		ASSERT(k_lower < k_higher);
		ASSERT(k_higher <= length());

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_higher; ++i) {
			if (__data[i] != T(0)) {
				return false;
			}
		}

		return true;
	}

	/// calculate the hamming weight
	/// \return the hamming weight
	[[nodiscard]] constexpr inline uint32_t popcnt(const uint32_t l = 0,
	                                               const uint32_t h = length()) const noexcept {
		uint32_t r = 0;

		LOOP_UNROLL();
		for (uint32_t i = l; i < h; ++i) {
			if (__data[i] > 0) {
				r += 1u;
			}
		}

		return r;
	}

	/// swap coordinate i, j, boundary checks are done
	/// \param i coordinate
	/// \param j coordinate
	constexpr void swap(const uint32_t i,
	                    const uint32_t j) noexcept {
		ASSERT(i < length() && j < length());
		SWAP(__data[i], __data[j]);
	}

	/// *-1
	/// \param i
	constexpr void flip(const uint32_t i) noexcept {
		ASSERT(i < length());
		ASSERT(__data[i] < q);
		__data[i] *= -1;
		__data[i] += q;
		__data[i] %= q;
	}

	/// mod operations
	/// NOTE: generic operation
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \return a%q, component wise
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline TT mod_T(const TT a) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (nr_bits * i)) & mask;
			c ^= (a1 % q) << (nr_bits * i);
		}

		/// note implicit call
		return c;
	}


	/// wt operations
	/// NOTE: generic operation, not really fast
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \return a%q, component wise
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline uint32_t popcnt_T(const TT a) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		uint32_t wt = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (nr_bits * i)) & mask;
			if (a1) {
				wt += 1u;
			}
		}

		/// note implicit call
		return wt;
	}

	/// neg operations
	/// NOTE: generic operation
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \return a%q, component wise
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline TT neg_T(const TT a) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (nr_bits * i)) & mask;
			c ^= ((q - a1) % q) << (nr_bits * i);
		}

		/// note implicit call
		return c;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a input
	/// \param b input
	/// \return a + b, component wise
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline TT add_T(const TT a,
	                                               const TT b) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a1 = (a >> (nr_bits * i)) & mask;
			const TT b1 = (b >> (nr_bits * i)) & mask;
			c ^= ((a1 + b1) % q) << (nr_bits * i);
		}

		/// note implicit call
		return c;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a - b
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline TT sub_T(const TT a,
	                                               const TT b) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (nr_bits * i)) & mask;
			const T b1 = (b >> (nr_bits * i)) & mask;
			c ^= ((a1 + (q - b1)) % q) << (nr_bits * i);
		}

		/// note implicit call
		return c;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a*b component wise
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline TT mul_T(const TT a,
	                                               const TT b) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (nr_bits * i)) & mask;
			const T b1 = (b >> (nr_bits * i)) & mask;
			c ^= ((a1 * b1) % q) << (nr_bits * i);
		}

		/// note implicit call
		return c;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a*b component wise
	template<typename TT = DataType>
	[[nodiscard]] constexpr static inline TT scalar_T(const TT a,
	                                                  const TT b) noexcept {
		static_assert(sizeof(TT) <= 16);
		static_assert(sizeof(TT) >= sizeof(T));
		constexpr uint32_t nr_limbs = sizeof(TT) / sizeof(T);
		constexpr uint32_t nr_bits = sizeof(T) * 8u;
		constexpr __uint128_t mask = (__uint128_t(1ull) << nr_bits) - 1ull;

		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (nr_bits * i)) & mask;
			c ^= ((a1 * b) % q) << (nr_bits * i);
		}

		/// note implicit call
		return c;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a+b, component wise
	[[nodiscard]] constexpr static inline S add256_T(const S a,
	                                                 const S b) noexcept {
		S ret;
		const T *a_data = (const T *) &a;
		const T *b_data = (const T *) &b;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < S::LIMBS; ++i) {
			ret_data[i] = (a_data[i] + b_data[i]) % q;
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a-b, component wise
	[[nodiscard]] constexpr static inline S sub256_T(const S a,
	                                                 const S b) noexcept {
		S ret;
		const T *a_data = (const T *) &a;
		const T *b_data = (const T *) &b;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < S::LIMBS; ++i) {
			ret_data[i] = (a_data[i] + (q - b_data[i])) % q;
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	[[nodiscard]] constexpr static inline S mul256_T(const S a,
	                                                 const S b) noexcept {
		S ret;
		const T *a_data = (const T *) &a;
		const T *b_data = (const T *) &b;
		T *ret_data = (T *)ret.d;
		for (uint8_t i = 0; i < S::LIMBS; ++i) {
			ret_data[i] = (a_data[i] * b_data[i]) % q;
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	[[nodiscard]] constexpr static inline S neg256_T(const S a) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		S ret;
		const T *a_data = (const T *) &a;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = (q - a_data[i]) % q;
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a
	/// \return a%q component wise
	[[nodiscard]] constexpr static inline S mod256_T(const S a) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint8x32_t ret;
		const T *data = (const T *) &a;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = data[i] % q;
		}

		return ret;
	}

	///  out[s: ] = in[0:s]
	constexpr static inline void sll(FqNonPackedVectorMeta &out,
									 const FqNonPackedVectorMeta &in,
									 const uint32_t s) noexcept {
		out.zero();

		ASSERT(s < length());
		for (uint32_t j = 0; j < length() - s; ++j) {
			const auto d = in.get(j);
			out.set(d, j + s);
		}
	}

	///
	constexpr static inline void slr(FqNonPackedVectorMeta &out,
									 const FqNonPackedVectorMeta &in,
									 const uint32_t s) noexcept {
		out.zero();

		ASSERT(s < length());
		for (uint32_t j = 0; j < length() - s; ++j) {
			const auto d = in.get(j+s);
			out.set(d, j);
		}
	}

	constexpr static inline void ror(FqNonPackedVectorMeta &out,
									 const FqNonPackedVectorMeta &in,
									 const uint32_t s) noexcept {
		out.zero();

		ASSERT(s < length());
		for (uint32_t j = 0; j < length(); ++j) {
			const auto d = in.get((j + s) % length());
			out.set(d, j);
		}
	}
	constexpr static inline void rol(FqNonPackedVectorMeta &out,
									 const FqNonPackedVectorMeta &in,
									 const uint32_t s) noexcept {
		out.zero();

		ASSERT(s < length());
		for (uint32_t j = 0; j < length() - s; ++j) {
			const auto d = in.get(j);
			out.set(d, (j + s) % length());
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(T *out, const T *in1) noexcept {
		uint32_t i = 0;
		for (; i + nr_of_limbs_in_S < n; i += nr_of_limbs_in_S) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t tmp = mod256_T(a);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mod_T<T>(in1[i]);
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(FqNonPackedVectorMeta &out,
	                                 const FqNonPackedVectorMeta &in1) noexcept {
		mod((T *) out.__data.data(), (const T *) in1.__data.data());
	}

	/// negate every coordinate between [k_lower, k_higher)
	/// \param k_lower lower dimension inclusive
	/// \param k_upper higher dimension, exclusive
	constexpr inline void neg(const uint32_t k_lower = 0,
	                          const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			__data[i] = (q - __data[i]) % q;
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void add(T *out,
	                                 const T *in1,
	                                 const T *in2) noexcept {
		uint32_t i = 0;
		for (; i + nr_of_limbs_in_S <= n; i += nr_of_limbs_in_S) {
			const S a = S::load((uint8_t *)(in1 + i));
			const S b = S::load((uint8_t *)(in2 + i));

			const S tmp = add256_T(a, b);
			S::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = add_T<T>(in1[i], in2[i]);
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void add(FqNonPackedVectorMeta &out,
	                       			 const FqNonPackedVectorMeta &in1,
	                       			 const FqNonPackedVectorMeta &in2) noexcept {
		add((T *) out.__data.data(),
		    (const T *) in1.ptr(),
		    (const T *) in2.ptr());
	}

	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \param norm = max norm of an dimension which is allowed.
	/// \return true if the element needs to be filtered out. False else.
	constexpr inline static bool add(FqNonPackedVectorMeta &v3,
	                                 FqNonPackedVectorMeta const &v1,
	                                 FqNonPackedVectorMeta const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper,
	                                 const uint32_t norm = -1) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		if (norm == -1u) {
			for (uint64_t i = k_lower; i < k_upper; ++i) {
				v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;
			}

			return false;
		}

		LOOP_UNROLL();
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;
			if (cryptanalysislib::math::abs(v3.__data[i]) > norm) {
				return true;
			}
		}

		return false;
	}

	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \param norm = max norm of an dimension which is allowed.
	/// \return true if the element needs to be filtered out. False else.
	template<const uint32_t k_lower, const uint32_t k_upper, const uint32_t norm=-1u>
	constexpr inline static bool add(FqNonPackedVectorMeta &v3,
	                                 FqNonPackedVectorMeta const &v1,
	                                 FqNonPackedVectorMeta const &v2) noexcept {
		static_assert( k_upper <= length() && k_lower < k_upper);

		if constexpr (norm == -1u) {
			for (uint64_t i = k_lower; i < k_upper; ++i) {
				v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;
			}

			return false;
		}

		LOOP_UNROLL();
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;
			if (cryptanalysislib::math::abs(v3.__data[i]) > norm){
				return true;
			}
		}

		return false;
	}
	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void sub(T *out,
	                       const T *in1,
	                       const T *in2) noexcept {
		uint32_t i = 0;
		for (; i + nr_of_limbs_in_S < n; i += nr_of_limbs_in_S) {
			const auto a = S::load((uint8_t *)(in1 + i));
			const auto b = S::load((uint8_t *)(in2 + i));

			const S tmp = sub256_T(a, b);
			S::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = sub_T<T>(in1[i], in2[i]);
		}
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void sub(FqNonPackedVectorMeta &out,
	                       const FqNonPackedVectorMeta &in1,
	                       const FqNonPackedVectorMeta &in2) noexcept {
		sub((T *) out.__data.data(),
		    (const T *) in1.__data.data(),
		    (const T *) in2.__data.data());
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \param norm filter every element out if hte norm is bigger than `norm`
	/// \return true if the elements needs to be filter out. False if not
	constexpr inline static bool sub(FqNonPackedVectorMeta &v3,
	                                 FqNonPackedVectorMeta const &v1,
	                                 FqNonPackedVectorMeta const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper,
	                                 const uint32_t norm = -1) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + q - v2.__data[i]) % q;
			if ((cryptanalysislib::math::abs(v3.__data[i]) > norm) && (norm != uint32_t(-1))) {
				return true;
			}
		}

		return false;
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \param norm filter every element out if hte norm is bigger than `norm`
	/// \return true if the elements needs to be filter out. False if not
	template<const uint32_t k_lower,
	         const uint32_t k_upper,
	         const uint32_t norm=-1u>
	constexpr inline static bool sub(FqNonPackedVectorMeta &v3,
	                                 FqNonPackedVectorMeta const &v1,
	                                 FqNonPackedVectorMeta const &v2) noexcept {
		static_assert(k_upper <= length() && k_lower < k_upper);

		if constexpr (norm == -1u){
			LOOP_UNROLL();
			for (uint32_t i = k_lower; i < k_upper; ++i) {
				v3.__data[i] = (v1.__data[i] + q - v2.__data[i]) % q;
			}

			return false;
		}

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + q - v2.__data[i]) % q;
			if (cryptanalysislib::math::abs(v3.__data[i]) > norm){
				return true;
			}
		}

		return false;
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void mul(T *out,
	                                 const T *in1,
	                       			 const T *in2,
	                                 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=n) noexcept {
		uint32_t i = k_lower;
		for (; i + nr_of_limbs_in_S <= k_upper; i += nr_of_limbs_in_S) {
			const auto a = S::load(in1 + i);
			const auto b = S::load(in2 + i);

			const S tmp = (S)mul256_T(a, b);
			S::store(out + i, tmp);
		}

		for (; i < k_upper; i += 1u) {
			out[i] = mul_T<T>(in1[i], in2[i]);
		}
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void mul(FqNonPackedVectorMeta &out,
	                       			 const FqNonPackedVectorMeta &in1,
	                       			 const FqNonPackedVectorMeta &in2,
	                                 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=n) noexcept {
		mul((T *) out.__data.data(),
		    (const T *) in1.__data.data(),
		    (const T *) in2.__data.data(),
		    k_lower, k_upper);
	}

	/// v1 = v1*v2 between [k_lower, k_upper)
	/// \param v1 input/output
	/// \param v2 input
	/// \param k_lower lower bound, inclusive
	/// \param k_upper upper bound, exclusive
	constexpr inline static void scalar(FqNonPackedVectorMeta &v1,
	                                    const DataType v2,
	                                    const uint32_t k_lower = 0,
	                                    const uint32_t k_upper = length()) noexcept {
		scalar(v1, v1, v2, k_lower, k_upper);
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	constexpr inline static void scalar(FqNonPackedVectorMeta &v3,
	                                    const FqNonPackedVectorMeta &v1,
	                                    const DataType v2,
	                                    const uint32_t k_lower = 0,
	                                    const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] * v2) % q;
		}
	}

	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \return v1 == v2 on the coordinates [k_lower, k_higher)
	constexpr inline static bool cmp(FqNonPackedVectorMeta const &v1,
	                                 FqNonPackedVectorMeta const &v2,
	                                 const uint32_t k_lower = 0,
	                                 const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		return true;
	}

	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \return v1 == v2 on the coordinates [k_lower, k_higher)
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline static bool cmp(FqNonPackedVectorMeta const &v1,
	                                 FqNonPackedVectorMeta const &v2) noexcept {
		static_assert( k_upper <= length() && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		return true;
	}

	/// static function.
	/// Sets v1 to v2 between [k_lower, k_higher). Does not touch the other
	/// coordinates in v1.
	/// \param v1 output container
	/// \param v2 input container
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	constexpr inline static void set(FqNonPackedVectorMeta &v1,
	                                 FqNonPackedVectorMeta const &v2,
	                                 const uint32_t k_lower = 0,
	                                 const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v1.__data[i] = v2.__data[i];
		}
	}

	/// \param obj to compare to
	/// \param k_lower lower coordinate bound, inclusive
	/// \param k_upper higher coordinate bound, exclusive
	/// \return this == obj on the coordinates [k_lower, k_higher)
	constexpr bool is_equal(FqNonPackedVectorMeta const &obj,
							const uint32_t k_lower = 0,
							const uint32_t k_upper = length()) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	/// \param obj to compare to
	/// \param k_lower lower coordinate bound, inclusive
	/// \param k_upper higher coordinate bound, exclusive
	/// \return this == obj on the coordinates [k_lower, k_higher)
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr bool is_equal(FqNonPackedVectorMeta const &obj) const noexcept {
		return cmp<k_lower, k_upper>(*this, obj);
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	/// \return this > obj on the coordinates [k_lower, k_higher)
	constexpr bool is_greater(FqNonPackedVectorMeta const &obj,
	                          const uint32_t k_lower = 0,
	                          const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length());
		ASSERT(k_lower < k_upper);
		
		LOOP_UNROLL();
		for (uint64_t i = k_upper; i > k_lower; i--) {
			if (__data[i - 1] > obj.__data[i - 1]) {
				return true;
			} else if (__data[i - 1] < obj.__data[i - 1]) {
				return false;
			}
		}
		// they are equal
		return false;
	}
	
	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	/// \return this > obj on the coordinates [k_lower, k_higher)
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr bool is_greater(FqNonPackedVectorMeta const &obj) const noexcept {
		static_assert(k_upper <= length());
		static_assert(k_lower < k_upper);

		LOOP_UNROLL();
		for (uint64_t i = k_upper; i > k_lower; i--) {
			if (__data[i - 1] > obj.__data[i - 1]) {
				return true;
			} else if (__data[i - 1] < obj.__data[i - 1]) {
				return false;
			}
		}
		// they are equal
		return false;
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	/// \return this < obj on the coordinates [k_lower, k_higher)
	constexpr bool is_lower(FqNonPackedVectorMeta const &obj,
	                        const uint32_t k_lower = 0,
	                        const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length());
		ASSERT(k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_upper; i > k_lower; i--) {
			if (__data[i - 1] < obj.__data[i - 1]) {
				return true;
			} else if (__data[i - 1] > obj.__data[i - 1]) {
				return false;
			}
		}

		return false;
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	/// \return this < obj on the coordinates [k_lower, k_higher)
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr bool is_lower(FqNonPackedVectorMeta const &obj) const noexcept {
		static_assert( k_upper <= length());
		static_assert( k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_upper; i > k_lower; i--) {
			if (__data[i - 1] < obj.__data[i - 1]) {
				return true;
			} else if (__data[i - 1] > obj.__data[i - 1]) {
				return false;
			}
		}

		return false;
	}

	///
	/// \tparam weight
	/// \tparam Ts
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	template<const uint32_t weight,
	         const uint32_t... Ts>
	constexpr static bool filter(FqNonPackedVectorMeta &v3,
	                             const FqNonPackedVectorMeta &v1,
	                             const FqNonPackedVectorMeta &v2,
	                             const uint32_t k_lower = 0,
	                             const uint32_t k_upper = n) {
		constexpr uint32_t nTs = sizeof...(Ts);
		static_assert(nTs == q);
		uint16_t ctr[nTs] = {0};
		uint32_t w = 0;

		for (uint64_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;

			/// weight check
			w += (v3.__data[i] != 0);
			if (w >= weight) {
				return false;
			}

			uint32_t j = 0;
			for (const auto p: {Ts...}) {
				if (v3.__data[i] == j) {
					ctr[j] += 1;
					if (ctr[j] >= p) {
						return false;
					}
				}

				j += 1;
			}
		}
		return true;
	}

	/// access operator
	/// \param i position. Boundary check is done.
	/// \return limb at position i
	[[nodiscard]] constexpr T &operator[](const size_t i) noexcept {
		ASSERT(i < length());
		return __data[i];
	}

	[[nodiscard]] constexpr const T &operator[](const size_t i) const noexcept {
		ASSERT(i < length());
		return __data[i];
	};

	/// prints this container in binary between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound, inclusive
	/// \param k_upper higher bound, exclusive
	constexpr void print_binary(const uint32_t k_lower = 0,
	                            const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_lower < length() && k_upper <= length() && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			unsigned data = (unsigned) __data[i];
			for (uint32_t j = 0; j < ceil_log2(q); ++j) {
				std::cout << (data & 1u) << " ";
				data >>= 1;
			}
		}
		std::cout << std::endl;
	}

	/// prints this container between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound, inclusive
	/// \param k_upper higher bound, exclusive
	constexpr void print(const uint32_t k_lower = 0,
	                     const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_lower < length() && k_upper <= length() && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << (unsigned) __data[i] << " ";
		}
		std::cout << std::endl;
	}

	/// iterators
	[[nodiscard]] constexpr inline auto begin() noexcept { return __data.begin(); }
	[[nodiscard]] constexpr inline auto begin() const noexcept { return __data.begin(); }
	[[nodiscard]] constexpr inline auto end() noexcept { return __data.end(); }
	[[nodiscard]] constexpr inline auto end() const noexcept { return __data.end(); }

	// this data container is never binary
	[[nodiscard]] __FORCEINLINE__ constexpr static bool binary() noexcept { return false; }
	[[nodiscard]] __FORCEINLINE__ constexpr static uint32_t size() noexcept { return length(); }
	[[nodiscard]] __FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return length(); }
	/// returns size of a single element in this container in bits
	[[nodiscard]] __FORCEINLINE__ constexpr static size_t sub_container_size() noexcept { return sizeof(T) * 8; }
	[[nodiscard]] __FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return length() * sizeof(T); }

	/// returns the underlying data container
	[[nodiscard]] __FORCEINLINE__ constexpr T *ptr() noexcept { return __data.data(); }
	[[nodiscard]] __FORCEINLINE__ constexpr const T *ptr() const noexcept { return __data.data(); }
	[[nodiscard]] __FORCEINLINE__ T ptr(const size_t i) noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};
	[[nodiscard]] const __FORCEINLINE__ T ptr(const size_t i) const noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};

	[[nodiscard]] __FORCEINLINE__ std::array<T, length()> &data() noexcept { return __data; }
	[[nodiscard]] __FORCEINLINE__ const std::array<T, length()> &data() const noexcept { return __data; }
	[[nodiscard]] constexpr T data(const size_t index) const noexcept {
		ASSERT(index < length());
		return __data[index];
	}
	[[nodiscard]] constexpr T get(const size_t index) const noexcept {
		ASSERT(index < length());
		return __data[index];
	}
	constexpr void set(const T data, const size_t index) noexcept {
		ASSERT(index < length());
		__data[index] = data % q;
	}

	/// sets all elements in the const_array to the given
	/// \param data
	constexpr void set(const T data) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < n; ++i) {
			__data[i] = data;
		}
	}

	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };

	///
	constexpr static void info() noexcept {
		std::cout << "{ name: \"kAryContainerMeta\""
		          << ", n: " << n
				  << ", q: " << q
				  << ", internal_limbs: " << internal_limbs
				  << ", sizeof(T): " << sizeof(T)
				  << "}" << std::endl;
	}
protected:
	std::array<T, length()> __data;
};


/// simple data container holding `length` Ts
/// \tparam T base type
/// \tparam length number of elements
/// \tparam q prime
template<const uint32_t n,
         const uint64_t q,
         typename T=uint64_t>
#if __cplusplus > 201709L
    requires kAryContainerAble<T> &&
             std::is_integral<T>::value
#endif
class FqNonPackedVector : public FqNonPackedVectorMeta<n, q, T> {
public:
	using M = FqNonPackedVectorMeta<n, q, T>;

	/// needed constants
	using M::length;
	using M::modulus;

	/// needed typedefs
	using typename M::DataType;
	using typename M::LimbType;
	using typename M::ContainerType;

	/// needed fields
	using M::__data;

	/// needed functions
	using M::get;
	using M::set;
	using M::random;
	using M::zero;
	using M::is_equal;
	using M::is_greater;
	using M::is_lower;
	using M::cmp;
	using M::print;
	using M::size;
	using M::data;
	using M::is_zero;

	/// needed arithmetic
	using M::mod;
	using M::neg;
	using M::add;
	using M::sub;
};

/// NOTE: this implements the representation padded (e.g. every number gets 8 bit)
/// NOTE: this is only the implementation for q=4
template<const uint32_t n>
#if __cplusplus > 201709L
    requires kAryContainerAble<uint8_t>
#endif
class FqNonPackedVector<n, 4, uint8_t > : public FqNonPackedVectorMeta<n, 4, uint8_t> {
public:
	/// this is just needed, because Im lazy
	constexpr static uint32_t q = 4;

	/// needed typedefs
	using T = uint8_t;
	using DataType = T;
	using M =  FqNonPackedVectorMeta<n, q, T>;

	using M::length;
	using M::modulus;
	using typename M::LimbType;
	using typename M::ContainerType;
	using M::__data;

	/// needed functions
	using M::get;
	using M::set;
	using M::random;
	using M::zero;
	using M::is_equal;
	using M::is_greater;
	using M::is_lower;
	using M::cmp;
	using M::print;
	using M::size;
	using M::data;
	using M::is_zero;
	using M::ptr;

	/// needed arith
	using M::neg;
	using M::mod;
	using M::add;
	using M::sub;

private:
	// helper masks
	static constexpr __uint128_t mask_4 = (__uint128_t(0x0303030303030303ULL) << 64UL) | (__uint128_t(0x0303030303030303ULL));
	static constexpr __uint128_t mask_q = (__uint128_t(0x0404040404040404ULL) << 64UL) | (__uint128_t(0x0404040404040404ULL));

public:

	/// mod operations
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \return a%q, component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T mod_T(const T a) noexcept {
		return a & mask_4;
	}

	/// mod operations
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \return a%q, component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T neg_T(const T a) noexcept {
		return (mask_q - a) & mask_4;
	}
	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a + b, component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T add_T(const T a,
	                                              const T b) noexcept {
		return (a + b) & mask_4;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a - b
	template<typename T>
	[[nodiscard]] constexpr static inline T sub_T(const T a,
	                                              const T b) noexcept {
		return (a - b + mask_q) & mask_4;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a*b component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T mul_T(const T a,
	                                              const T b) noexcept {
		constexpr uint32_t nr_limbs = sizeof(T);
		constexpr __uint128_t mask = 0xf;
		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (8u * i)) & mask;
			const T b1 = (b >> (8u * i)) & mask;
			c ^= (a1 * b1) & mask_4;
		}

		/// note implicit call
		return c;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a
	/// \return a%q component wise
	constexpr static inline uint8x32_t mod256_T(const uint8x32_t a) noexcept {
		const uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		return uint8x32_t::and_(a, mask256_4);
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a+b, component wise
	constexpr static inline uint8x32_t add256_T(const uint8x32_t a,
	                                            const uint8x32_t b) noexcept {
		constexpr uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		return uint8x32_t::and_(uint8x32_t::add(a, b), mask256_4);
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a-b, component wise
	constexpr static inline uint8x32_t sub256_T(const uint8x32_t a,
	                                            const uint8x32_t b) noexcept {
		constexpr uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		constexpr uint8x32_t mask256_q = uint8x32_t::set1(0x04);
		return uint8x32_t::and_(uint8x32_t::add(uint8x32_t::sub(a, b), mask256_q), mask256_4);
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	static inline uint8x32_t mul256_T(const uint8x32_t a,
	                                  const uint8x32_t b) noexcept {
		constexpr uint8x32_t mask256_3 = uint8x32_t::set1(0x03);
		const uint8x32_t tmp = uint8x32_t::mullo(a, b);
		return uint8x32_t::and_(tmp, mask256_3);
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(uint8_t *out,
	                          const uint8_t *in1,
	                          const T in2) noexcept {
		ASSERT(in2 <= q);
		uint32_t i = 0;

		const uint8x32_t b = uint8x32_t::set1(in2);

		LOOP_UNROLL()
		for (; i + 32 < n; i += 32) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		LOOP_UNROLL()
		for (; i < n; i += 1) {
			out[i] = mul_T<uint8_t>(in1[i], in2);
		}
	}

	/// scalar multiplication
	/// \param out output = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: scalar
	template<typename T>
	static inline void scalar(FqNonPackedVector &out,
	                          const FqNonPackedVector &in1,
	                          const T in2) noexcept {
		scalar<T>(out.__data.data(), in1.__data.data(), in2);
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(T *out,
	                                 const T *in1) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t tmp = mod256_T(a);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mod_T<T>(in1[i]);
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(FqNonPackedVector &out,
	                                 const FqNonPackedVector &in1) noexcept {
		mod((T *) out.__data.data(), (const T *) in1.__data.data());
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(T *out,
	                       const T *in1,
	                       const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = add256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = add_T<T>(in1[i], in2[i]);
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		add((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(T *out,
	                       const T *in1,
	                       const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs <= n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = sub256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = sub_T<T>(in1[i], in2[i]);
		}
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		sub((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void mul(T *out,
	                       			 const T *in1,
	                       			 const T *in2,
	                                 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=n) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = k_lower;
		for (; i + nr_limbs <= k_upper; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < k_upper; i += 1) {
			out[i] = mul_T<T>(in1[i], in2[i]);
		}
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	constexpr static inline void mul(FqNonPackedVector &out,
	                       			 const FqNonPackedVector &in1,
	                       			 const FqNonPackedVector &in2,
	                                 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=n) noexcept {
		mul((T *) out.__data.data(),
		    (const T *) in1.__data.data(),
		    (const T *) in2.__data.data(),
		    k_lower, k_upper);
	}

	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };
};


/// NOTE: this implements the representation padded (e.g. every number gets 8 bit)
/// NOTE: this is only the implementation for q=5
template<const uint32_t n>
#if __cplusplus > 201709L
    requires kAryContainerAble<uint8_t>
#endif
class FqNonPackedVector<n, 5, uint8_t > : public FqNonPackedVectorMeta<n, 5, uint8_t > {
public:
	/// this is just needed, because Im lazy
	constexpr static uint32_t q = 5;

	/// needed typedefs
	using T = uint8_t;
	using DataType = T;
	using M = FqNonPackedVectorMeta<n, q, T>;

	using M::length;
	using M::modulus;
	using typename M::LimbType;
	using typename M::ContainerType;
	using M::__data;

	/// needed functions
	using M::get;
	using M::set;
	using M::random;
	using M::zero;
	using M::is_equal;
	using M::is_greater;
	using M::is_lower;
	using M::cmp;
	using M::print;
	using M::size;
	using M::data;
	using M::is_zero;
	using M::ptr;

	/// needed arith
	using M::neg;
	using M::mod;
	using M::add;
	using M::sub;


private:
	// helper masks
	static constexpr __uint128_t mask_7 = (__uint128_t(0x0707070707070707ULL) << 64UL) | (__uint128_t(0x0707070707070707ULL));
	static constexpr __uint128_t mask_5 = (__uint128_t(0x0505050505050505ULL) << 64UL) | (__uint128_t(0x0505050505050505ULL));
	static constexpr __uint128_t mask_3 = (__uint128_t(0x0303030303030303ULL) << 64UL) | (__uint128_t(0x0303030303030303ULL));
	static constexpr __uint128_t mask_f = (__uint128_t(0x0f0f0f0f0f0f0f0fULL) << 64UL) | (__uint128_t(0x0f0f0f0f0f0f0f0fULL));

public:
	/// NOTE: only works if each limb < 25
	/// mod operations
	/// STC: http://homepage.cs.uiowa.edu/~dwjones/bcd/mod.shtml
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a input
	/// \return a%5, component wise on uint8_t
	template<typename T>
	static T mod_T(const T aa) noexcept {
		static_assert(sizeof(T) <= 16);
		T a = aa;
		a = (3 * ((a >> 3) & mask_f)) + (a & mask_7);
		a = (3 * ((a >> 3) & mask_7)) + (a & mask_7);
		const T mask1 = ((a + mask_3) >> 3) & mask_7;
		const T mask2 = (mask1 & mask_5) ^ ((mask1 << 2u) & mask_5);
		a = a - mask2;
		return a;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a + b, component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T add_T(const T a, const T b) noexcept {
		return mod_T(a + b);
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a - b
	template<typename T>
	[[nodiscard]] constexpr static inline T sub_T(const T a, const T b) noexcept {
		return mod_T(a + (mask_5 - b));
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a MUST be reduced
	/// \param b MUST be reduces
	/// \return a*b component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T mul_T(const T a, const T b) noexcept {
		constexpr uint32_t nr_limbs = sizeof(T);
		const __uint128_t mask = 0xf;
		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (8u * i)) & mask;
			const T b1 = (b >> (8u * i)) & mask;
			c ^= mod_T(a1 * b1) << (8u * i);
		}

		return c;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a
	/// \return a%q component wise
	constexpr static inline uint8x32_t mod256_T(const uint8x32_t aa) noexcept {
		constexpr uint8x32_t mask256_3 = uint8x32_t::set1(0x03);
		constexpr uint8x32_t mask256_5 = uint8x32_t::set1(0x05);
		constexpr uint8x32_t mask256_7 = uint8x32_t::set1(0x07);
		constexpr uint8x32_t mask256_f = uint8x32_t::set1(0x0f);
		uint8x32_t a = aa;
		a = (3 * ((a >> 3) & mask256_f)) + (a & mask256_7);
		a = (3 * ((a >> 3) & mask256_7)) + (a & mask256_7);
		const uint8x32_t mask1 = ((a + mask256_3) >> 3) & mask256_7;
		const uint8x32_t mask2 = (mask1 & mask256_5) ^ ((mask1 << 2u) & mask256_5);
		a = a - mask2;
		return a;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a+b, component wise
	static inline uint8x32_t add256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		return mod256_T(uint8x32_t::add(a, b));
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a-b, component wise
	static inline uint8x32_t sub256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		const uint8x32_t mask256_5 = uint8x32_t::set1(0x05);
		return mod256_T(uint8x32_t::add(a, uint8x32_t::sub(mask256_5, b)));
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	static inline uint8x32_t mul256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		return mod256_T(uint8x32_t::mullo(a, b));
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(uint8_t *out, const uint8_t *in1, const T in2) noexcept {
		uint32_t i = 0;

		const uint8x32_t b = uint8x32_t::set1(in2);
		for (; i + 32 < n; i += 32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mul_T<uint8_t>(in1[i], in2);
		}
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(FqNonPackedVector &out, const FqNonPackedVector &in1, const T in2) noexcept {
		scalar<T>(out.__data.data(), in1.__data.data(), in2);
	}


	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(T *out, const T *in1) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t tmp = mod256_T(a);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mod_T<T>(in1[i]);
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(FqNonPackedVector &out,
	                                 const FqNonPackedVector &in1) noexcept {
		mod((T *) out.__data.data(), (const T *) in1.__data.data());
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(T *out, const T *in1, const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = add256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = add_T<T>(in1[i], in2[i]);
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		add((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	template<const uint32_t k_lower,
			 const uint32_t k_upper,
			 const uint32_t norm>
	static inline void add(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		add((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(T *out, const T *in1, const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = sub256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = sub_T<T>(in1[i], in2[i]);
		}
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		sub((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(T *out, const T *in1, const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mul_T<T>(in1[i], in2[i]);
		}
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		mul((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };
};


/// NOTE: this implements the representation padded (e.g. every number gets 8 bit)
/// NOTE: this is only the implementation for q=7
template<const uint32_t n>
#if __cplusplus > 201709L
    requires kAryContainerAble<uint8_t>
#endif
class FqNonPackedVector<n, 7, uint8_t > : public FqNonPackedVectorMeta<n, 7, uint8_t> {
public:
	/// this is just needed, because Im lazy
	constexpr static uint32_t q = 7;

	/// needed typedefs
	using T = uint8_t;
	using DataType = T;
	using M = FqNonPackedVectorMeta<n, q, T>;

	using M::length;
	using M::modulus;
	using typename M::LimbType;
	using typename M::ContainerType;
	using M::__data;

	/// needed functions
	using M::get;
	using M::set;
	using M::random;
	using M::zero;
	using M::is_equal;
	using M::is_greater;
	using M::is_lower;
	using M::cmp;
	using M::print;
	using M::size;
	using M::data;
	using M::is_zero;
	using M::ptr;

	/// needed arith
	using M::neg;
	using M::mod;
	using M::add;
	using M::sub;

private:
	// helper masks
	static constexpr __uint128_t mask_1 = (__uint128_t(0x0101010101010101ULL) << 64UL) | (__uint128_t(0x0101010101010101ULL));
	static constexpr __uint128_t mask_7 = (__uint128_t(0x0707070707070707ULL) << 64UL) | (__uint128_t(0x0707070707070707ULL));

public:
	/// NOTE: only works if each limb < 49
	/// mod operations
	/// STC: http://homepage.cs.uiowa.edu/~dwjones/bcd/mod.shtml
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \return a%q, component wise on uint8_t
	template<typename T>
	[[nodiscard]] constexpr static inline T mod_T(const T aa) noexcept {
		static_assert(sizeof(T) <= 16);
		T a = aa;
		//a = ((a >> 4) & mask_f) + (a & mask_f);
		//a = ((a >> 3) & mask_f) + (a & mask_7);
		a = ((a >> 3) & mask_7) + (a & mask_7);
		a = ((a >> 3) & mask_7) + (a & mask_7);
		a = ((((a + mask_1) >> 3) & mask_7) + a) & mask_7;
		return a;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a + b, component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T add_T(const T a, const T b) noexcept {
		return mod_T(a + b);
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a - b
	template<typename T>
	[[nodiscard]] constexpr static inline T sub_T(const T a, const T b) noexcept {
		return mod_T(a + (mask_7 - b));
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a MUST be reduced
	/// \param b MUST be reduces
	/// \return a*b component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T mul_T(const T a, const T b) noexcept {
		constexpr uint32_t nr_limbs = sizeof(T);
		const __uint128_t mask = 0xf;
		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (8u * i)) & mask;
			const T b1 = (b >> (8u * i)) & mask;
			c ^= mod_T(a1 * b1) << (8u * i);
		}

		return c;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a
	/// \return a%q component wise
	static inline uint8x32_t mod256_T(const uint8x32_t aa) noexcept {
		constexpr uint8x32_t mask256_7 = uint8x32_t::set1(0x07);
		constexpr uint8x32_t mask256_1 = uint8x32_t::set1(0x01);
		uint8x32_t a = aa;
		a = ((a >> 3) & mask256_7) + (a & mask256_7);
		a = ((a >> 3) & mask256_7) + (a & mask256_7);
		a = ((((a + mask256_1) >> 3) & mask256_7) + a) & mask256_7;
		return a;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a+b, component wise
	static inline uint8x32_t add256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		return mod256_T(uint8x32_t::add(a, b));
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a-b, component wise
	static inline uint8x32_t sub256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		const uint8x32_t mask256_7 = uint8x32_t::set1(0x07);
		return mod256_T(uint8x32_t::add(a, uint8x32_t::sub(mask256_7, b)));
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	static inline uint8x32_t mul256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		return mod256_T(uint8x32_t::mullo(a, b));
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(uint8_t *out, const uint8_t *in1, const T in2) noexcept {
		uint32_t i = 0;

		const uint8x32_t b = uint8x32_t::set1(in2);
		for (; i + 32 < n; i += 32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mul_T<uint8_t>(in1[i], in2);
		}
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(FqNonPackedVector &out, const FqNonPackedVector &in1, const T in2) noexcept {
		scalar<T>(out.__data.data(), in1.__data.data(), in2);
	}


	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(T *out, const T *in1) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t tmp = mod256_T(a);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mod_T<T>(in1[i]);
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vector
	constexpr static inline void mod(FqNonPackedVector &out,
	                                 const FqNonPackedVector &in1) noexcept {
		mod((T *) out.__data.data(), (const T *) in1.__data.data());
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(T *out, const T *in1, const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = add256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = add_T<T>(in1[i], in2[i]);
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		add((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(T *out, const T *in1, const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = sub256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = sub_T<T>(in1[i], in2[i]);
		}
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		sub((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(T *out, const T *in1, const T *in2) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint32_t i = 0;
		for (; i + nr_limbs < n; i += nr_limbs) {
			const uint8x32_t a = uint8x32_t::load(in1 + i);
			const uint8x32_t b = uint8x32_t::load(in2 + i);

			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::store(out + i, tmp);
		}

		for (; i < n; i += 1) {
			out[i] = mul_T<T>(in1[i], in2[i]);
		}
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(FqNonPackedVector &out,
	                       const FqNonPackedVector &in1,
	                       const FqNonPackedVector &in2) noexcept {
		mul((T *) out.__data.data(), (const T *) in1.__data.data(), (const T *) in2.__data.data());
	}

	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };
};


///
/// \tparam T
/// \tparam n
/// \tparam q
/// \param a
/// \param b
/// \return
template<const uint32_t n, const uint64_t q, typename T>
constexpr inline bool operator==(const FqNonPackedVectorMeta<n, q, T> &a,
								 const FqNonPackedVectorMeta<n, q, T> &b) noexcept {
	return a.is_equal(b);
}

///
/// \tparam n
/// \tparam q
/// \tparam T
/// \param a
/// \param b
/// \return
template<const uint32_t n, const uint64_t q, typename T>
constexpr inline bool operator<(const FqNonPackedVectorMeta<n, q, T> &a,
								const FqNonPackedVectorMeta<n, q, T> &b) noexcept {
	return a.is_lower(b);
}

///
/// \tparam n
/// \tparam q
/// \tparam T
/// \param a
/// \param b
/// \return
template<const uint32_t n, const uint64_t q, typename T>
constexpr inline bool operator>(const FqNonPackedVectorMeta<n, q, T> &a,
								const FqNonPackedVectorMeta<n, q, T> &b) noexcept {
	return a.is_greater(b);
}

///
/// \tparam T
/// \tparam n
/// \tparam q
/// \param out
/// \param obj
/// \return
template<const uint32_t n, const uint64_t q, typename T>
std::ostream &operator<<(std::ostream &out, const FqNonPackedVector<n, q, T> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << unsigned(obj[i]);
	}
	return out;
}
#endif
