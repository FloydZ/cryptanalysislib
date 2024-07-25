#ifndef SMALLSECRETLWE_FQ_PACKED_VECTOR_H
#define SMALLSECRETLWE_FQ_PACKED_VECTOR_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "container/binary_packed_vector.h"
#include "helper.h"
#include "math/math.h"
#include "popcount/popcount.h"
#include "random.h"
#include "simd/simd.h"


#if defined(USE_AVX2)
#include <immintrin.h>
#endif

#if __cplusplus > 201709L
/// Concept for the base data type
/// \tparam T
template<typename T>
concept kAryPackedContainerAble =
        std::is_integral<T>::value && requires(T t) {
	        t ^ t;
	        t + t;
        };
#endif

/// represents a vector of numbers mod `MOD` in vector of `T` in a compressed way
/// Meta class, contains all important meta definitions.
/// \param T = uint64_t
/// \param n = number of elements
/// \param q = modulus
template<class T, 
		 const uint64_t _n,
		 const uint64_t _q>
#if __cplusplus > 201709L
    requires std::is_integral<T>::value
#endif
class kAryPackedContainer_Meta {
public:
	typedef kAryPackedContainer_Meta ContainerType;

	// make the length and modulus of the container public available
	constexpr static uint64_t q = _q;
	constexpr static inline uint64_t modulus() { return q; }
	constexpr static uint32_t n = _n;
	constexpr static inline uint64_t length() { return n; }
	
	static_assert(n > 0);

	// number of bits in each T
	constexpr static uint16_t bits_per_limb = sizeof(T) * 8;
	// number of bits needed to represent MOD
	constexpr static uint32_t bits_per_number = (uint32_t) bits_log2(q);
	static_assert(bits_per_number > 0);
	// number of numbers one can fit into each limb
	constexpr static uint16_t numbers_per_limb = (bits_per_limb + bits_per_number - 1) / bits_per_number;
	static_assert(numbers_per_limb > 0);
	// Number of Limbs needed to represent `length` numbers of size log(MOD) +1
	constexpr static uint16_t internal_limbs = (n + numbers_per_limb - 1) / numbers_per_limb;
	// mask with the first `bits_per_number` set to one
	constexpr static T number_mask = (T(1u) << (bits_per_number)) - 1u;

	// true if we need every bit of the last bit
	constexpr static bool is_full = (n % bits_per_limb) == 0;

	constexpr static bool activate_avx2 = true;
	constexpr static uint16_t limbs_per_simd_limb = 256u / bits_per_limb;
	constexpr static uint16_t numbers_per_simd_limb = 256u / bits_per_number;

	constexpr static uint32_t total_bits = bits_per_limb * internal_limbs;
	constexpr static uint32_t total_bytes = sizeof(T) * internal_limbs;

	// we are good C++ devs.
	typedef T ContainerLimbType;
	using DataType = LogTypeTemplate<bits_per_number>;

	// list compatibility typedef
	typedef T LimbType;
	typedef T LabelContainerType;


	// internal data
	std::array<T, internal_limbs> __data;

	// simple hash function
	constexpr inline uint64_t hash() const noexcept {
		return __data[0];
	}

	/// the mask is only valid for one internal number.
	/// \param i bit position the read
	/// \return bit mask to access the i-th element within a limb
	constexpr inline T accessMask(const uint16_t i) const noexcept {
		return number_mask << (i % numbers_per_limb);
	}

	// round a given amount of 'in' bits to the nearest limb excluding the lowest overflowing bits
	// eg 13 -> 64
	constexpr static uint16_t round_up(uint16_t in) noexcept { return round_up_to_limb(in) * bits_per_limb; }
	constexpr static uint16_t round_up_to_limb(uint16_t in) noexcept { return (in / bits_per_limb) + 1; }

	// the same as above only rounding down
	// 13 -> 0
	constexpr static uint16_t round_down(uint16_t in) noexcept { return round_down_to_limb(in) * bits_per_limb; }
	constexpr static uint16_t round_down_to_limb(uint16_t in) { return (in / bits_per_limb); }

	// given the i-th bit this function will return a bits mask where the lower 'i' bits are set. Everything will be
	// realigned to limb_bits_width().
	constexpr static T lower_mask(const uint16_t i) noexcept {
		ASSERT(i < n);
		return ((T(1) << (i % bits_per_limb)) - 1);
	}

	// given the i-th bit this function will return a bits mask where the higher (n-i)bits are set.
	constexpr static T higher_mask(const uint16_t i) noexcept {
		ASSERT(i < n);
		if ((i % bits_per_limb) == 0) return T(-1);

		return (~((T(1u) << (i % bits_per_limb)) - 1));
	}
	/// access the i-th coordinate/number/element
	/// \param i coordinate to access.
	/// \return the number you wanted to access, shifted down to the lowest bits.
	constexpr inline DataType get(const uint16_t i) const noexcept {
		// needs 5 instructions. So 64*5 for the whole limb
		ASSERT(i < length());
		return DataType((__data[i / numbers_per_limb] >> ((i % numbers_per_limb) * bits_per_number)) & number_mask);
	}

	/// sets the `i`-th number to `data`
	/// \param data value to set the const_array n
	/// \param i -th number to overwrite
	/// \return nothing
	constexpr inline void set(const DataType data, const uint32_t i) noexcept {
		ASSERT(i < length());
		const uint16_t off = i / numbers_per_limb;
		const uint16_t spot = (i % numbers_per_limb) * bits_per_number;

		T bla = (number_mask & T(data % modulus())) << spot;
		T val = __data[off] & (~(number_mask << spot));
		__data[off] = bla | val;
	}

	/// sets the `i`-th number to `data`
	/// \param data value to set the const_array on
	/// \param i -th number to overwrite
	/// \return nothing
	constexpr void set(const DataType data) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < n; ++i) {
			set(data, i);
		}
	}

	/// set everything to zero
	constexpr void zero() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < internal_limbs; i++) {
			__data[i] = 0;
		}
	}

	/// sets everything to zero between [a, b)
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void zero(const uint32_t a, const uint32_t b) noexcept {
		for (uint32_t i = a; i < b; i++) {
			set(0, i);
		}
	}

	/// \return nothing
	constexpr inline void clear() noexcept {
		zero();
	}

	/// set everything to one
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void one(const uint32_t a = 0, const uint32_t b = length()) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(1, i);
		}
	}

	/// Set everything to two
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void two(const uint32_t a = 0, const uint32_t b = length()) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(2 % modulus(), i);
		}
	}

	/// generates all limbs uniformly random
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void random(const uint32_t a = 0, const uint32_t b = length()) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(fastrandombytes_uint64() % modulus(), i);
		}
	}

	/// computes the hamming weight
	[[nodiscard]] constexpr uint32_t popcnt() const noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < size(); i++) {
			ret += get(i) > 0;
		}

		return ret;
	}

	/// return the positions of the first p bits/numbers set
	/// \param out output: const_array of the first p positions set in the container
	/// \param p maximum bits!=0 to find
	constexpr void get_bits_set(uint16_t *out, const uint32_t p) const noexcept {
		uint32_t ctr = 0;
		for (uint32_t i = 0; i < length(); i++) {
			if (unsigned(get(i)) != 0u) {
				out[ctr] = i;
				ctr += 1u;
			}

			// early exit
			if (ctr == p) {
				return;
			}
		}
	}

	/// \return true if every limb is empty
	[[nodiscard]] constexpr bool is_zero() const noexcept {
		for (uint32_t i = 0; i < internal_limbs; ++i) {
			if (__data[i] != 0)
				return false;
		}

		return true;
	}

	/// checks if every number between [a, b) is empyt
	/// \param a lower bound
	/// \param b higher bound
	/// \return true/false
	[[nodiscard]] constexpr bool is_zero(const uint32_t a, const uint32_t b) const noexcept {
		for (uint32_t i = a; i < b; ++i) {
			if (get(i) != 0)
				return false;
		}

		return true;
	}

	/// swap the numbers on positions `i`, `j`.
	/// \param i first coordinate
	/// \param j second coordinate
	constexpr void swap(const uint16_t i, const uint16_t j) noexcept {
		ASSERT(i < length() && j < length());
		auto tmp = get(i);
		set(i, get(j));
		set(j, tmp);
	}


	/// infix negates (x= -x mod q)  all numbers between [k_lower, k_higher)
	/// \param k_lower lower limit, inclusive
	/// \param k_upper higher limit, exclusive
	constexpr inline void neg(const uint32_t k_lower = 0, const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			set(((-get(i)) + modulus()) % modulus(), i);
		}
	}

	/// negates (x = -q mod q) betweet [k_lower, k_upper)
	/// \tparam k_lower inclusive
	/// \tparam k_upper exclusive
	template<uint32_t k_lower, uint32_t k_upper>
	constexpr inline void neg() noexcept {
		for (uint32_t i = k_lower; i < k_upper; i++) {
			set(((-get(i)) + modulus()) % modulus(), i);
		}
	}

	/// NOTE: generic implementation
	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static TT add_T(const TT in1, const TT in2) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			const TT b = (in2 >> (bits_per_number * i)) & mask;
			ret ^= ((a + b) % q) << (bits_per_number * i);
		}
		return ret;
	}

	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static TT sub_T(const TT in1, const TT in2) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			const TT b = (in2 >> (bits_per_number * i)) & mask;
			ret ^= ((a - b + q) % q) << (bits_per_number * i);
		}
		return ret;
	}

	/// NOTE: generic implementation
	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static TT mul_T(const TT in1, const TT in2) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			const TT b = (in2 >> (bits_per_number * i)) & mask;
			ret ^= ((a * b) % q) << (bits_per_number * i);
		}
		return ret;
	}

	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static TT mod_T(const TT in1) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			ret ^= (a % q) << (bits_per_number * i);
		}
		return ret;
	}

	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static TT neg_T(const TT in1) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			ret ^= ((q - a) % q) << (bits_per_number * i);
		}
		return ret;
	}

	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static uint32_t popcnt_T(const TT in1) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		uint32_t ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			ret += a > 0;
		}

		return ret;
	}

	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT = DataType>
	constexpr inline static TT scalar_T(const TT in1, const TT in2) noexcept {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT) * 8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			ret ^= ((a * in2) % q) << (bits_per_number * i);
		}
		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a+b, component wise
	[[nodiscard]] constexpr static inline uint8x32_t add256_T(const uint8x32_t a,
	                                                          const uint8x32_t b) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint8x32_t ret;
		const T *a_data = (const T *) &a;
		const T *b_data = (const T *) &b;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = add_T(a_data[i], b_data[i]);
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a-b, component wise
	[[nodiscard]] constexpr static inline uint8x32_t sub256_T(const uint8x32_t a,
	                                                          const uint8x32_t b) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint8x32_t ret;
		const T *a_data = (const T *) &a;
		const T *b_data = (const T *) &b;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = sub_T(a_data[i], b_data[i]);
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	[[nodiscard]] constexpr static inline uint8x32_t mul256_T(const uint8x32_t a,
	                                                          const uint8x32_t b) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint8x32_t ret;
		const T *a_data = (const T *) &a;
		const T *b_data = (const T *) &b;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = mul_T(a_data[i], b_data[i]);
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	[[nodiscard]] constexpr static inline uint8x32_t neg256_T(const uint8x32_t a) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint8x32_t ret;
		const T *a_data = (const T *) &a;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = neg_T(a_data[i]);
		}

		return ret;
	}

	/// helper function
	/// vectorized version
	/// \param a
	/// \return a%q component wise
	[[nodiscard]] constexpr static inline uint8x32_t mod256_T(const uint8x32_t a) noexcept {
		constexpr uint32_t nr_limbs = 32u / sizeof(T);

		uint8x32_t ret;
		const T *a_data = (const T *) &a;
		T *ret_data = (T *) &ret;
		for (uint8_t i = 0; i < nr_limbs; ++i) {
			ret_data[i] = neg_T(a_data[i]);
		}

		return ret;
	}

	/// v1 += v1
	/// \param v2 input/output
	/// \param v1 input
	constexpr inline static void add(kAryPackedContainer_Meta &v1,
	                                 kAryPackedContainer_Meta const &v2) noexcept {
		add(v1, v1, v2, 0, length());
	}

	/// v3 = v1 + v2
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	constexpr inline static void add(kAryPackedContainer_Meta &v3,
	                                 kAryPackedContainer_Meta const &v1,
	                                 kAryPackedContainer_Meta const &v2) noexcept {
		add(v3, v1, v2, 0, length());
	}

	/// generic add: v3 = v1 + v2 between [k_lower, k_upper)
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower limit inclusive
	/// \param k_upper upper limit exclusive
	constexpr inline static void add(kAryPackedContainer_Meta &v3,
	                                 kAryPackedContainer_Meta const &v1,
	                                 kAryPackedContainer_Meta const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = v1.get(i) + v2.get(i);
			v3.set(data % modulus(), i);
		}
	}

	/// v1 -= v2
	/// \param v1 input/output
	/// \param v2 input
	constexpr inline static void sub(kAryPackedContainer_Meta &v1,
	                                 kAryPackedContainer_Meta const &v2) noexcept {
		sub(v1, v1, v2, 0, length());
	}

	/// v3 = v1 - v2 mod q
	/// NOTE: this computes the subtraction on all coordinates
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	constexpr inline static void sub(kAryPackedContainer_Meta &v3,
	                                 kAryPackedContainer_Meta const &v1,
	                                 kAryPackedContainer_Meta const &v2) noexcept {
		sub(v3, v1, v2, 0, length());
	}

	/// v3 = v1 - v2 between [k_lower, k_upper)
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	constexpr inline static void sub(kAryPackedContainer_Meta &v3,
	                                 kAryPackedContainer_Meta const &v1,
	                                 kAryPackedContainer_Meta const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			int64_t data = int64_t(v1.get(i)) - int64_t(v2.get(i));
			if (data < 0)
				data += modulus();
			v3.set(data % modulus(), i);
		}
	}

	/// generic components mul: v3 = v1 * v2 between [k_lower, k_upper)
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower limit inclusive
	/// \param k_upper upper limit exclusive
	constexpr inline static void mul(kAryPackedContainer_Meta &v3,
	                                 kAryPackedContainer_Meta const &v1,
	                                 kAryPackedContainer_Meta const &v2,
	                                 const uint32_t k_lower = 0,
	                                 const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = (v1.get(i) * v2.get(i)) % modulus();
			v3.set(data, i);
		}
	}

	/// generic components mod: v3 = v1 %q between [k_lower, k_upper)
	/// \param v3 output
	/// \param v1 input
	/// \param k_lower lower limit inclusive
	/// \param k_upper upper limit exclusive
	constexpr inline static void mod(kAryPackedContainer_Meta &v3,
	                                 kAryPackedContainer_Meta const &v1,
	                                 const uint32_t k_lower = 0,
	                                 const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = v1.get(i) % modulus();
			v3.set(data, i);
		}
	}
	/// generic components mul: v3 = v1 * v2 between [k_lower, k_upper)
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower limit inclusive
	/// \param k_upper upper limit exclusive
	template<class TT = DataType>
	constexpr inline static void scalar(kAryPackedContainer_Meta &v3,
	                                    kAryPackedContainer_Meta const &v1,
	                                    const TT v2,
	                                    const uint32_t k_lower = 0,
	                                    const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = (v1.get(i) * v2) % modulus();
			v3.set(data, i);
		}
	}

	/// \param v1 input
	/// \param v2 input
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	/// \return v1 == v2 between [k_lower, k_upper)
	constexpr inline static bool cmp(kAryPackedContainer_Meta const &v1,
	                                 kAryPackedContainer_Meta const &v2,
	                                 const uint32_t k_lower = 0,
	                                 const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			if (v1.get(i) != v2.get(i))
				return false;
		}
		return true;
	}

	/// v1 = v2 between [k_lower, k_upper)
	/// \param v1 output
	/// \param v2 input
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	constexpr inline static void set(kAryPackedContainer_Meta &v1,
	                                 kAryPackedContainer_Meta const &v2,
	                                 const uint32_t k_lower = 0,
	                                 const uint32_t k_upper = length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			v1.set(v2.get(i), i);
		}
	}

	/// \param obj
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	/// \return this == obj between [k_lower, k_upper)
	constexpr bool is_equal(kAryPackedContainer_Meta const &obj,
	                        const uint32_t k_lower = 0,
	                        const uint32_t k_upper = length()) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	///
	/// \param obj
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	/// \return this > obj [k_lower, k_upper)
	constexpr bool is_greater(kAryPackedContainer_Meta const &obj,
	                          const uint32_t k_lower = 0,
	                          const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_upper; i > k_lower; i--) {
			if (get(i - 1) < obj.get(i - 1))
				return false;
		}

		return true;
	}

	///
	/// \param obj
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	/// \return this < obj [k_lower, k_upper)
	constexpr bool is_lower(kAryPackedContainer_Meta const &obj,
	                        const uint32_t k_lower = 0,
	                        const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_upper; i > k_lower; i--) {
			if (get(i - 1) > obj.get(i - 1))
				return false;
		}
		return true;
	}

	/// add on full length and return the weight only between [l, h)
	/// \param v3 = v1 + v2
	/// \param v1 input
	/// \param v2 input
	/// \param l lower limit
	/// \param h upper limit
	/// \return weight of v3 between [l, h)
	constexpr static uint16_t add_only_upper_weight_partly(kAryPackedContainer_Meta &v3,
	                                                       const kAryPackedContainer_Meta &v1,
	                                                       const kAryPackedContainer_Meta &v2,
	                                                       const uint32_t l, const uint32_t h) noexcept {
		uint16_t weight = 0;
		add(v3, v1, v2);
		for (uint32_t i = l; i < h; ++i) {
			weight += v3.get(i) != 0;
		}

		return weight;
	}

	/// shifts this by i to the left
	/// \param i amount to shift
	constexpr void left_shift(const uint32_t i) noexcept {
		ASSERT(i < length());
		for (uint32_t j = length(); j > i; j--) {
			set(get(j - i - 1), j - 1);
		}

		// clear the rest.
		for (uint32_t j = 0; j < i; j++) {
			set(0, j);
		}
	}

	/// shifts this by i to the left
	/// \param i amount to shift
	constexpr void right_shift(const uint32_t i) noexcept {
		ASSERT(i < length());
		for (uint32_t j = 0; j < n - i; j--) {
			const auto data = get(i + j);
			set(data, j);
		}

		// clear the rest.
		for (uint32_t j = 0; j < i; j++) {
			set(0, n - j - 1u);
		}
	}

	/// prints between [k_lower, k_upper )
	/// \param k_lower lower limit
	/// \param k_upper upper limit
	void print(const uint32_t k_lower = 0, const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_lower < length() && k_upper <= length() && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << unsigned(get(i));
		}
		std::cout << "\n";
	}

	/// print the `pos
	/// \param data
	/// \param k_lower inclusive
	/// \param k_higher exclusive
	void static print_binary(const uint64_t data,
	                         const uint16_t k_lower = 0,
	                         const uint16_t k_higher = length()) noexcept {
		uint64_t d = data;
		for (uint32_t i = k_lower; i < k_higher; ++i) {
			std::cout << unsigned(d & 1);
			std::cout << unsigned((d >> 1) & 1);
			d >>= 2;
		}
		std::cout << "\n";
	}

	/// print the `pos
	/// \param k_lower inclusive
	/// \param k_higher exclusive
	void print_binary(const uint32_t k_lower = 0, const uint32_t k_higher = length()) const noexcept {
		for (uint32_t i = k_lower; i < k_higher; ++i) {
			std::cout << unsigned(get(i) & 1);
			std::cout << unsigned((get(i) >> 1) & 1);
		}
		std::cout << "\n";
	}

	// iterators.
	auto begin() noexcept { return __data.begin(); }
	auto begin() const noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }
	auto end() const noexcept { return __data.end(); }

	///
	/// \param i
	/// \return
	constexpr DataType operator[](size_t i) noexcept {
		ASSERT(i < length());
		return get(i);
	}

	///
	/// \param i
	/// \return
	constexpr DataType operator[](const size_t i) const noexcept {
		ASSERT(i < length());
		return get(i);
	};

	// return `true` if the datastruct contains binary data.
	__FORCEINLINE__ constexpr static bool binary() noexcept { return false; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return length(); }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return internal_limbs; }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return internal_limbs * sizeof(T); }

	std::array<T, internal_limbs> &data() noexcept { return __data; }
	const std::array<T, internal_limbs> &data() const noexcept { return __data; }

	T &data(const size_t index) {
		ASSERT(index < length());
		return __data[index];
	}
	const T data(const size_t index) const noexcept {
		ASSERT(index < length());
		return __data[index];
	}
	T limb(const size_t index) {
		ASSERT(index < length());
		return __data[index];
	}
	const T limb(const size_t index) const noexcept {
		ASSERT(index < length());
		return __data[index];
	}

	// get raw access to the underlying data.
	T *ptr() noexcept { return __data.data(); }
	const T *ptr() const noexcept { return __data.data(); }
	__FORCEINLINE__ T ptr(const size_t i) noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};
	const __FORCEINLINE__ T ptr(const size_t i) const noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};
	// returns `false` as this class implements a generic arithmetic
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return false; };
};

/// represents a vector of numbers mod `MOD` in vector of `T` in a compressed way
/// \param T = uint64_t
/// \param n = number of elemtns
/// \param q = modulus
template<class T, const uint32_t n, const uint32_t q>
#if __cplusplus > 201709L
    requires kAryPackedContainerAble<T> &&
             std::is_integral<T>::value
#endif
class kAryPackedContainer_T : public kAryPackedContainer_Meta<T, n, q> {
	/// Nomenclature:
	///     Number 	:= actual data one wants to save % modulus()
	///		Limb 	:= Underlying data container holding at max sizeof(T)/log2(modulus()) many numbers.
	/// Its not possible, that numbers cover more than one limb.
	/// The internal data container layout looks like this:
	/// 		limb0				limb1			limb2
	///   [	n0	,  n1  ,  n2  |	    , 	,	  |		,	 ,     |  .... ]
	/// The container fits as much numbers is the limb as possible. But there will no overhanging elements (e.g.
	///  numbers that first bits are on one limb and the remaining bits are on the next limb).
	///

public:
	using kAryPackedContainer_Meta<T, n, q>::bits_per_limb;
	using kAryPackedContainer_Meta<T, n, q>::bits_per_number;
	using kAryPackedContainer_Meta<T, n, q>::numbers_per_limb;
	using kAryPackedContainer_Meta<T, n, q>::internal_limbs;
	using kAryPackedContainer_Meta<T, n, q>::number_mask;
	using kAryPackedContainer_Meta<T, n, q>::is_full;
	using kAryPackedContainer_Meta<T, n, q>::activate_avx2;

	/// some function
	using kAryPackedContainer_Meta<T, n, q>::mod_T;
	using kAryPackedContainer_Meta<T, n, q>::sub_T;
	using kAryPackedContainer_Meta<T, n, q>::add_T;
	using kAryPackedContainer_Meta<T, n, q>::mul_T;


	using typename kAryPackedContainer_Meta<T, n, q>::ContainerLimbType;
	using typename kAryPackedContainer_Meta<T, n, q>::LimbType;
	using typename kAryPackedContainer_Meta<T, n, q>::LabelContainerType;
	// minimal internal datatype to present an element.
	using DataType = LogTypeTemplate<bits_per_number>;

	using kAryPackedContainer_Meta<T, n, q>::length;
	using kAryPackedContainer_Meta<T, n, q>::modulus;
	using kAryPackedContainer_Meta<T, n, q>::__data;

public:
	// base constructor
	constexpr kAryPackedContainer_T() = default;
};

/// lel, C++ metaprogramming is king
/// \tparam n
template<const uint32_t n>
class kAryPackedContainer_T<uint64_t, n, 2> : public BinaryContainer<n, uint64_t> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 2;
};

///
/// partly specialized class for q=3
template<const uint32_t n>
#if __cplusplus > 201709L
    requires kAryPackedContainerAble<uint64_t> &&
             std::is_integral<uint64_t>::value
#endif
class kAryPackedContainer_T<uint64_t, n, 3> : public kAryPackedContainer_Meta<uint64_t, n, 3> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;
	using T = uint64_t;

	/// needed size descriptions
	using kAryPackedContainer_Meta<T, n, q>::bits_per_limb;
	using kAryPackedContainer_Meta<T, n, q>::bits_per_number;
	using kAryPackedContainer_Meta<T, n, q>::numbers_per_limb;
	using kAryPackedContainer_Meta<T, n, q>::internal_limbs;
	using kAryPackedContainer_Meta<T, n, q>::number_mask;
	using kAryPackedContainer_Meta<T, n, q>::is_full;
	using kAryPackedContainer_Meta<T, n, q>::activate_avx2;
	using kAryPackedContainer_Meta<T, n, q>::limbs_per_simd_limb;
	using kAryPackedContainer_Meta<T, n, q>::numbers_per_simd_limb;

	/// needed type definitions
	using typename kAryPackedContainer_Meta<T, n, q>::ContainerLimbType;
	using typename kAryPackedContainer_Meta<T, n, q>::LimbType;
	using typename kAryPackedContainer_Meta<T, n, q>::LabelContainerType;
	// minimal internal datatype to present an element.
	using DataType = LogTypeTemplate<bits_per_number>;

	/// needed
	using kAryPackedContainer_Meta<T, n, q>::length;
	using kAryPackedContainer_Meta<T, n, q>::modulus;
	using kAryPackedContainer_Meta<T, n, q>::__data;

	// extremly important
	typedef kAryPackedContainer_T<T, n, q> ContainerType;

	/// some functions
	using kAryPackedContainer_Meta<T, n, q>::get;
	using kAryPackedContainer_Meta<T, n, q>::set;
	using kAryPackedContainer_Meta<T, n, q>::is_equal;
	using kAryPackedContainer_Meta<T, n, q>::is_greater;
	using kAryPackedContainer_Meta<T, n, q>::is_lower;
	using kAryPackedContainer_Meta<T, n, q>::add;

public:
	/// calculates the hamming weight of one limb.
	/// IMPORTANT: only correct if there is no 3 in one of the limbs
	/// \param a
	/// \return
	template<typename TT = DataType>
	static inline uint16_t popcnt_T(const TT a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr TT c1 = sizeof(TT) == 16 ? (TT(6148914691236517205u) << 64u) | TT(6148914691236517205u) : TT(6148914691236517205u);
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT c2 = sizeof(TT) == 16 ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);

		const TT ac1 = a & c1;// filter the ones
		const TT ac2 = a & c2;// filter the twos

		return cryptanalysislib::popcount::template popcount<TT>(ac1) +
		       cryptanalysislib::popcount::template popcount<TT>(ac2);
	}

	/// computes the hamming weight of a ternary vector
	/// \param a input
	/// \return a mod 3, in every coordinate
	static inline uint16_t popcnt256_T(const uint64x4_t a) noexcept {
		constexpr static uint64x4_t c1 = uint64x4_t::set1(6148914691236517205u);
		constexpr static uint64x4_t c2 = uint64x4_t::set1(12297829382473034410u);
		const uint64x4_t ac1 = c1 & a;// filter the ones
		const uint64x4_t ac2 = c2 & a;// filter the twos
		const uint64x4_t t = uint64x4_t::popcnt(ac1) + uint64x4_t::popcnt(ac2);
		return t.v64[0] + t.v64[1] + t.v64[2] + t.v64[3];
	}

	/// \param lower lower bound, inclusive
	/// \param upper upper bound, exclusive
	constexpr inline void neg(const uint32_t lower,
	                          const uint32_t upper) noexcept {
		ASSERT(lower <= upper);
		ASSERT(upper <= n);

		// NOTE: its important that its signed
		const int32_t lower_limb = (lower + bits_per_limb - 1u) / bits_per_limb;
		const int32_t upper_limb = (upper + bits_per_limb - 1u) / bits_per_limb;
		for (int32_t i = lower_limb + 1; i < upper_limb - 1; i++) {
			__data[i] = neg_T(__data[i]);
		}

		for (uint32_t i = lower; i < (lower_limb * numbers_per_limb); ++i) {
			const uint32_t data = (q - get(i)) % q;
			set(data, i);
		}

		for (uint32_t i = (upper_limb * numbers_per_limb); i < upper; ++i) {
			const uint32_t data = (q - get(i)) % q;
			set(data, i);
		}
	}

	/// negates the vector on all coordinates
	constexpr inline void neg() noexcept {
		if constexpr (internal_limbs == 1) {
			__data[0] = neg_T(__data[0]);
			return;
		}

		uint32_t i = 0;

		for (; i + 2 <= internal_limbs; i += 2) {
			__uint128_t t = neg_T<__uint128_t>(*((__uint128_t *) &__data[i]));
			*((__uint128_t *) &__data[i]) = t;
		}

		for (; i < internal_limbs; i++) {
			__data[i] = neg_T(__data[i]);
		}
	}

	/// \tparam k_lower lower limit inclusive
	/// \tparam k_upper upper limit exclusive
	template<uint32_t k_lower, uint32_t k_upper>
	constexpr inline void neg() noexcept {
		static_assert(k_upper <= length() && k_lower < k_upper);

		constexpr uint32_t ll = 2 * k_lower / bits_per_limb;
		constexpr uint32_t lh = 2 * k_upper / bits_per_limb;
		constexpr uint32_t ol = 2 * k_lower % bits_per_limb;
		constexpr uint32_t oh = 2 * k_upper % bits_per_limb;
		constexpr T ml = ~((T(1) << ol) - T(1));
		constexpr T mh = (T(1) << oh) - T(1);

		constexpr T nml = ~ml;
		constexpr T nmh = ~mh;

		if constexpr (ll == lh) {
			constexpr T m = ml & mh;
			constexpr T nm = ~m;
			__data[ll] = (neg_T(__data[ll]) & m) ^ (__data[ll] & nm);
		} else {
			for (uint32_t i = ll + 1; i < lh - 1; ++i) {
				__data[i] = neg_T(__data[i]);
			}
			__data[ll] = (neg_T(__data[ll]) & ml) ^ (__data[ll] & nml);
			__data[lh] = (neg_T(__data[lh]) & mh) ^ (__data[lh] & nmh);
		}
	}

	/// negates one limb
	/// \param a input limb
	/// \return negative limb
	template<typename TT = DataType>
	constexpr static inline TT neg_T(const TT a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr TT c1 = sizeof(TT) == 16 ? (TT(6148914691236517205u) << 64u) | TT(6148914691236517205u) : TT(6148914691236517205u);
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT c2 = sizeof(TT) == 16 ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);

		const TT e1 = a & c1;// filter the ones
		const TT e2 = a & c2;// filter the twos

		// re-shifts everything to the correct place
		return (e1 << 1u) ^ (e2 >> 1u);
	}


	/// negate a on every coordinate
	/// \param a input
	/// \return -a
	static inline uint64x4_t neg256_T(const uint64x4_t a) noexcept {
		constexpr static uint64x4_t c1 = uint64x4_t::set1(6148914691236517205u);
		constexpr static uint64x4_t c2 = uint64x4_t::set1(12297829382473034410u);

		const uint64x4_t e1 = a & c1;
		const uint64x4_t e2 = a & c2;

		const uint64x4_t e11 = e1 << 1u;
		const uint64x4_t e21 = e2 >> 1u;

		return e11 ^ e21;
	}

	/// calculates a mod 3 on everything number
	template<typename TT = DataType>
	constexpr static inline TT mod_T(const TT a) noexcept {
		const T e = mod_T_withoutcorrection<TT>(a);

		// fix the overflow on coordinate 32
		constexpr TT ofmask = (T(1u) << 62) - 1;
		const TT ofbit = ((a >> 62) % 3) << 62;
		return (e & ofmask) ^ ofbit;
	}

	/// same as mod3_limb but without the correction of the last entry
	/// \param a
	/// \return
	template<typename TT = DataType>
	constexpr static inline TT mod_T_withoutcorrection(const TT a) noexcept {
		// int(0b1100110011001100110011001100110011001100110011001100110011001100)
		constexpr TT f = sizeof(TT) == 16 ? (TT(14757395258967641292u) << 64u) | TT(14757395258967641292u) : TT(14757395258967641292u);
		// int(0b001100110011001100110011001100100110011001100110011001100110011)
		constexpr TT g = sizeof(TT) == 16 ? (TT(3689348814741910323u) << 64u) | TT(3689348814741910323u) : TT(3689348814741910323u);
		// int(0b0100010001000100010001000100010001000100010001000100010001000100)
		constexpr TT c1 = sizeof(TT) == 16 ? (TT(4919131752989213764u) << 64u) | TT(4919131752989213764u) : TT(4919131752989213764u);
		// int(0b0001000100010001000100010001000100010001000100010001000100010001)
		constexpr TT c2 = sizeof(TT) == 16 ? (TT(1229782938247303441u) << 64u) << TT(1229782938247303441u) : TT(1229782938247303441u);
		const TT c = a & f;
		const TT d = a & g;

		const TT cc = ((c + c1) >> 2) & f;// adding one to simulate the carry bit
		const TT dc = ((d + c2) >> 2) & g;

		const TT cc2 = c + cc;
		const TT dc2 = d + dc;

		const TT cf = cc2 & f;// filter out again resulting carry bits
		const TT dg = dc2 & g;
		const TT e = (cf ^ dg);

		return e;
	}

	/// \param x input number
	/// \param y output number
	/// \return x-y in every number
	template<typename TT = DataType>
	constexpr static inline TT sub_T(const TT x, const TT y) noexcept {
		return add_T(x, neg_T(y));
	}

	/// \param x ; input
	/// \param y ; input
	/// \return x - y
	static inline uint64x4_t sub256_T(const uint64x4_t x,
	                                  const uint64x4_t y) noexcept {
		return add256_T(x, neg256_T(y));
	}

	/// v3 = v1 - v2
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower
	/// \param k_upper
	constexpr inline static void sub(kAryPackedContainer_T &v3,
	                                 kAryPackedContainer_T const &v1,
	                                 kAryPackedContainer_T const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			int64_t data = int64_t(v1.get(i)) - int64_t(v2.get(i)) + modulus();
			v3.set(data % modulus(), i);
		}
	}

	///
	/// \param v3 output: = v1 - v2
	/// \param v1 input:
	/// \param v2 input:
	constexpr inline static void sub(kAryPackedContainer_T &v3,
	                                 kAryPackedContainer_T const &v1,
	                                 kAryPackedContainer_T const &v2) noexcept {
		//if constexpr (internal_limbs == 1) {
		//	v3.__data[0] = sub_T(v1.__data[0], v2.__data[0]);
		//	return;
		//} else if constexpr ((internal_limbs == 2) && (sizeof(T) == 8)) {
		//	__uint128_t t = sub_T<__uint128_t>(*((__uint128_t *)v1.__data.data()), *((__uint128_t *)v2.__data.data()));
		//	*((__uint128_t *)v3.__data.data()) = t;
		//	return;
		//}

		uint32_t i = 0;// TODO not working on ARM apple
		//if constexpr(activate_avx2) {
		//	for (; i + limbs_per_simd_limb <= internal_limbs; i += limbs_per_simd_limb) {
		//		const uint64x4_t t = sub256_T(uint64x4_t::unaligned_load((uint64x4_t *) &v1.__data[i]),
		//		                              uint64x4_t::unaligned_load((uint64x4_t *) &v2.__data[i]));
		//		uint64x4_t::unaligned_store((uint64x4_t *) &v3.__data[i], t);
		//	}
		//}

		for (; i < internal_limbs; i++) {
			v3.__data[i] = sub_T(v1.__data[i], v2.__data[i]);
		}
	}

	/// NOTE: this function, does not reduce anything.
	/// only call it if you know, that now overflow happens
	/// \tparam TT
	/// \param a input
	/// \param b input
	/// \return a+b
	template<typename TT>
	constexpr static inline TT add_T_no_overflow(const TT a,
	                                             const TT b) noexcept {
		// we know that no overflow will happen, So no 2+2=1
		return a + b;
	}

	/// \param x input first number
	/// \param y input second number
	/// \return x+y in every number
	template<typename TT>
	constexpr static inline TT add_T(const TT x, const TT y) noexcept {
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT c2 = sizeof(TT) == 16u ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr TT c1 = sizeof(TT) == 16u ? (TT(6148914691236517205u) << 64u) | TT(6148914691236517205u) : TT(6148914691236517205u);

		// These are not the optimal operations to calculate the ternary addition. But nearly.
		// The problem is that one needs to spit the limb for the ones/two onto two separate limbs. But two separate
		// limbs mean
		//      - higher memory consumption for each container
		//      - complicated hashing for the hashmaps
		const TT xy = x ^ y;
		const TT xy2 = x & y;
		const TT a = xy & c1;
		const TT b = xy & c2;
		const TT c = xy2 & c1;
		const TT d = xy2 & c2;
		const TT e = a & (b >> 1);

		const TT r0 = e ^ (d >> 1) ^ (a);
		const TT r1 = (e << 1) ^ (b) ^ (c << 1);
		return r0 ^ r1;
	}

	/// this function assumes that a,b \in [0,1], so there is no 2.
	/// \param a input
	/// \param b input
	/// \return a+b
	constexpr static inline uint64x4_t add256_T_no_overflow(const uint64x4_t a,
	                                                        const uint64x4_t b) noexcept {
		// no overflow will happen.
		return a + b;
	}

	/// \param x input
	/// \param y input
	/// \return x+y mod3
	static inline uint64x4_t add256_T(const uint64x4_t x,
	                                  const uint64x4_t y) noexcept {
		constexpr static uint64x4_t c1 = uint64x4_t::set1(6148914691236517205ull);
		constexpr static uint64x4_t c2 = uint64x4_t::set1(12297829382473034410ull);

		const uint64x4_t xy = x ^ y;
		const uint64x4_t xy2 = x & y;
		const uint64x4_t a = xy & c1;
		const uint64x4_t b = xy & c2;
		const uint64x4_t c = xy2 & c1;
		const uint64x4_t d = xy2 & c2;

		const uint64x4_t e = a & (b >> 1u);

		const uint64x4_t r0 = e ^ (d >> 1u) ^ a;
		const uint64x4_t r1 = b ^ (e << 1u) ^ (c << 1u);
		return r0 ^ r1;
	}

	/// \param v3 output = v1 + v2 mod3
	/// \param v1 input
	/// \param v2 input
	constexpr inline static void add(kAryPackedContainer_T &v3,
	                                 kAryPackedContainer_T const &v1,
	                                 kAryPackedContainer_T const &v2) noexcept {
		if constexpr (internal_limbs == 1) {
			v3.__data[0] = add_T(v1.__data[0], v2.__data[0]);
			return;
		} else if constexpr ((internal_limbs == 2) && (sizeof(DataType) == 8)) {
			const __uint128_t t = add_T<__uint128_t>(*((__uint128_t *) v1.__data.data()), *((__uint128_t *) v2.__data.data()));
			*(__uint128_t *) v3.__data.data() = t;
			return;
		} else if constexpr ((internal_limbs == 4) && (sizeof(DataType) == 8u)) {
			const uint64x4_t t = add256_T(uint64x4_t::aligned_load((uint64_t *) &v1.__data[0]),
			                              uint64x4_t::aligned_load((uint64_t *) &v2.__data[0]));
			uint64x4_t::unaligned_store((uint64x4_t *) &v3.__data[0], t);
			return;
		}

		uint32_t i = 0;
		if constexpr (activate_avx2) {
			for (; i + numbers_per_limb <= internal_limbs; i += numbers_per_simd_limb) {
				const uint64x4_t t = add256_T(uint64x4_t::unaligned_load((uint64_t *) &v1.__data[i]),
				                              uint64x4_t::unaligned_load((uint64_t *) &v2.__data[i]));
				uint64x4_t::unaligned_store((uint64x4_t *) &v3.__data[i], t);
			}
		}

		for (; i < internal_limbs; i++) {
			v3.__data[i] = add_T(v1.__data[i], v2.__data[i]);
		}
	}

	/// optimised version of the function above.
	/// \tparam l lower limit, inclusive
	/// \tparam h upper limit, exclusive
	/// \param v3 output = v1 + v2 on the full length and weight between [l, h)
	/// \param v1 input
	/// \param v2 input
	/// \return hamming weight
	template<const uint32_t l, const uint32_t h>
	constexpr static uint16_t add_only_weight_partly(kAryPackedContainer_T &v3,
	                                                 kAryPackedContainer_T &v1,
	                                                 kAryPackedContainer_T &v2) noexcept {
		constexpr uint32_t llimb = l / numbers_per_limb;
		constexpr uint32_t hlimb = h / numbers_per_limb;
		constexpr T lmask = ~((T(1u) << (l * bits_per_number)) - 1);
		constexpr T hmask = (T(1u) << (h * bits_per_number)) - 1;
		uint16_t weight = 0;

		// first add the lower limbs
		for (uint32_t i = 0; i < llimb; i++) {
			v3.__data[i] = add_T(v1.__data[i], v2.__data[i]);
		}

		// add the limb with weight
		v3.__data[llimb] = add_T(v1.__data[llimb], v2.__data[llimb]);
		weight += popcnt_T(v3.__data[llimb] & lmask);

		// add the limbs between l and h
		for (uint32_t i = llimb + 1; i < hlimb; i++) {
			v3.__data[i] = add_T(v1.__data[i], v2.__data[i]);
			weight += popcnt_T(v3.__data[i]);
		}

		// add the high limb
		v3.__data[hlimb] = add_T(v1.__data[hlimb], v2.__data[hlimb]);
		weight += popcnt_T(v3.__data[hlimb] & hmask);

		// add everything that is left
		for (uint32_t i = hlimb + 1; i < internal_limbs; i++) {
			v3.__data[i] = add_T(v1.__data[i], v2.__data[i]);
		}

		return weight;
	}

	/// \param a input
	/// \return 2*a, input must be reduced mod 3
	template<typename TT = DataType>
	constexpr static inline TT times2_T(const TT a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr TT t1 = sizeof(TT) == 16u ? (TT(6148914691236517205u) << 64u) | TT(6148914691236517205u) : TT(6148914691236517205u);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT t2 = sizeof(TT) == 16u ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);

		const TT tm1 = a & t1;              // extract the ones
		const TT tm2 = a & t2;              // extract the twos
		const TT acc = ((tm1 << 1u) ^ tm2); // where are not zeros
		const TT b = add_T<TT>(a, t2 & acc);// add two
		return b ^ (tm1 << 1u);
	}

	/// \param a element to check.
	/// \return true if `a` contains a two at any coordinate
	template<typename TT = DataType>
	constexpr static inline bool filter2_T(const TT a) noexcept {
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT m = sizeof(TT) == 16 ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);
		return (a & m) != 0;
	}

	/// \param a element to check
	/// \param limit
	/// \return returns the number of two in the limb
	template<typename TT = DataType>
	constexpr static inline uint32_t filter2count_T(const TT a) noexcept {
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT m = sizeof(TT) == 16u ? (TT(12297829382473034410ull) << 64u) | TT(12297829382473034410ull) : TT(12297829382473034410ull);
		return cryptanalysislib::popcount::template popcount<TT>(a & m);
	}

	/// \tparam k_lower lower limit in coordinates to check if twos exist
	/// \tparam k_upper upper limit in coordinates (not bits)
	/// \param a element to check if two exists
	/// \param limit how many twos are in total allowed
	/// \return return the twos in a[k_lower, k_upper].
	template<const uint16_t k_lower, const uint16_t k_upper, typename TT = DataType>
	constexpr static inline uint32_t filter2count_range_T(const TT a) noexcept {
		static_assert(k_lower != 0 && k_lower < k_upper && k_upper <= length());
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT m = sizeof(TT) == 16u ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);
		constexpr TT mask = ((TT(1u) << (2u * k_lower)) - 1u) & ((TT(1u) << (2u * k_upper)) - 1u);
		return cryptanalysislib::popcount::template popcount<TT>(a & mask & m);
	}

	/// counts the number of twos upto `k_upper` (exclusive)
	/// \tparam kupper
	template<const uint16_t k_upper, typename TT = DataType>
	constexpr inline uint32_t filter2count_T() {
		static_assert(k_upper <= length());
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT m = sizeof(TT) == 16 ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);
		constexpr TT mask = (TT(1u) << (2u * k_upper) % bits_per_limb) - 1u;
		constexpr uint32_t limb = std::max(1, (k_upper + numbers_per_limb - 1) / numbers_per_limb);

		if constexpr (limb == 1) {
			return cryptanalysislib::popcount::template popcount<TT>(__data[0] & mask & m);
		}

		uint32_t ctr = 0;
#pragma unroll
		for (uint32_t i = 0; i < limb - 1; ++i) {
			ctr += cryptanalysislib::popcount::template popcount<TT>(__data[i] & m);
		}

		return ctr + cryptanalysislib::popcount::template popcount<TT>(__data[limb - 1] & mask & m);
	}

	/// counts the number of twos in a single limb
	/// upto `k_upper` (exclusive)
	/// \tparam k_upper upper limit
	/// \param a input
	/// \return number of twos
	template<const uint16_t k_upper, typename TT = DataType>
	constexpr static inline uint32_t filter2count_range_T(const TT a) noexcept {
		ASSERT(0 < k_upper && k_upper <= length());
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr TT m = sizeof(TT) == 16u ? (TT(12297829382473034410u) << 64u) | TT(12297829382473034410u) : TT(12297829382473034410u);
		constexpr TT mm = (TT(1u) << (2u * k_upper)) - 1u;
		constexpr TT mask = m & mm;

		return cryptanalysislib::popcount::template popcount<TT>(a & mask);
	}

	/// checks whether an element needs be filtered or not.
	/// \param limit number of two allowed in the container.
	/// \return true if the element contains more than `limit` twos
	///			false otherwise
	constexpr inline bool filter2_mod3(const uint32_t limit) const noexcept {
		uint32_t ctr = 0;
		for (uint32_t i = 0; i < internal_limbs; ++i) {
			ctr += filter2count_mod3_limb(__data[i]);
			if (ctr > limit)
				return true;
		}

		return false;
	}

	/// v3 = 2*v1
	/// \param v3 output
	/// \param v1 input
	constexpr static inline void times2_mod3(kAryPackedContainer_T &v3, const kAryPackedContainer_T &v1) noexcept {
		uint32_t i = 0;
		for (; i < internal_limbs; i++) {
			v3.__data[i] = times2_T(v1.__data[i]);
		}
	}


	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };
};


///
/// \tparam T
/// \tparam n
/// \tparam q
/// \param out
/// \param obj
/// \return
template<typename T, const uint32_t n, const uint32_t q>
std::ostream &operator<<(std::ostream &out, const kAryPackedContainer_T<T, n, q> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << unsigned(obj[i]);
	}
	return out;
}
#endif
