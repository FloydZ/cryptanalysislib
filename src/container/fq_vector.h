#ifndef SMALLSECRETLWE_FQ_VECTOR_H
#define SMALLSECRETLWE_FQ_VECTOR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <atomic>

#include "helper.h"
#include "random.h"
#include "simd/avx2.h"
#include "simd/simd.h"
#include "popcount/popcount.h"

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

/// Concept for base data type.
/// \tparam T
template<typename T>
concept kAryContainerAble =
requires(T t) {
	t ^ t;
	T(0);
};


///
/// \tparam T base type
/// \tparam n size of the fq vector space=number of elements
/// \tparam q field size
template<typename T, const uint32_t n, const uint32_t q>
#if __cplusplus > 201709L
requires std::is_integral<T>::value
#endif
class kAryContainerMeta {
public:
	// internal data length. Used in the template system to pass through this information
	constexpr static uint32_t LENGTH = n;
	constexpr static uint32_t MODULUS = q;
	constexpr static uint16_t internal_limbs = n;

	// Needed for the internal template system.
	typedef T DataType;
	typedef T ContainerLimbType;

	typedef kAryContainerMeta ContainerType;

	/// zeros our the whole container
	/// \return nothing
	constexpr inline void zero() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < LENGTH; i++){
			__data[i] = T(0);
		}
	}

	/// zeros the data
	/// \return nothing
	constexpr inline void clear() noexcept {
		zero();
	}

	/// set everything one
	/// \return nothing
	constexpr inline void one() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < LENGTH; i++) {
			__data[i] = T(1);
		}
	}

	/// generates random coordinates
	/// \param k_lower lower coordinate to start from
	/// \param k_higher higher coordinate to stop. Not included.
	void random(uint32_t k_lower=0, uint32_t k_higher=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_higher; i++){
			__data[i] = fastrandombytes_uint64() % q;
		}
	}

	/// checks if every dimension is zero
	/// \return true/false
	[[nodiscard]] constexpr bool is_zero() const noexcept {
		for (uint32_t i = 0; i < LENGTH; ++i) {
			if(__data[i] != T(0))
				return false;
		}

		return true;
	}

	/// calculate the hamming weight
	/// \return the hamming weight
	[[nodiscard]] constexpr inline uint32_t weight(const uint32_t l=0, const uint32_t h=LENGTH) const noexcept {
		uint32_t r = 0;

		LOOP_UNROLL();
		for (uint32_t i = l; i < h; ++i) {
			if (__data[i] > 0)
				r += 1u;
		}

		return r;
	}

	/// swap coordinate i, j, boundary checks are done
	/// \param i coordinate
	/// \param j coordinate
	constexpr void swap(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < LENGTH && j < LENGTH);
		SWAP(__data[i], __data[j]);
	}

	/// *-1
	/// \param i
	constexpr void flip(const uint32_t i) noexcept {
		ASSERT(i < LENGTH);
		ASSERT(__data[i] < q);
		__data[i] *= -1;
		__data[i] += q;
		__data[i] %= q;
	}

	/// negate every coordinate between [k_lower, k_higher)
	/// \param k_lower lower dimension inclusive
	/// \param k_upper higher dimension, exclusive
	constexpr inline void neg(const uint32_t k_lower=0,
	                          const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH &&k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			__data[i] = (q - __data[i]) % q;
		}
	}

	/// v1 = v1*v2 between [k_lower, k_upper)
	/// \param v1 output/input
	/// \param v2 input
	/// \param k_lower lower bound: inclusive
	/// \param k_upper upper bound: exclusive
	constexpr inline static void mul(kAryContainerMeta &v1,
									 const kAryContainerMeta &v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=LENGTH) noexcept {
		mul(v1, v1, v2, k_lower, k_upper);
	}

	/// component wise multiplication
	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	constexpr inline static void mul(kAryContainerMeta &v3,
						    		 const kAryContainerMeta &v1,
						    		 const kAryContainerMeta &v2,
						    		 const uint32_t k_lower=0,
						    		 const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] * v2.__data[i]) % q;
		}
	}

	/// v1 = v1*v2 between [k_lower, k_upper)
	/// \param v1 input/output
	/// \param v2 input
	/// \param k_lower lower bound, inclusive
	/// \param k_upper upper bound, exclusive
	constexpr inline static void scalar(kAryContainerMeta &v1,
									 const DataType v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=LENGTH) noexcept {
		scalar(v1, v1, v2, k_lower, k_upper);
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	constexpr inline static void scalar(kAryContainerMeta &v3,
									 const kAryContainerMeta &v1,
									 const DataType v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] * v2) % q;
		}
	}

	/// v1 = v1 + v2, between [k_lower, k_upper)
	/// \param v1 output/input
	/// \param v2 input
	/// \param k_lower lower bound: inclusive
	/// \param k_upper upper bound: exclusive
	constexpr inline static void add(kAryContainerMeta &v1,
									 const kAryContainerMeta &v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=LENGTH) noexcept {
		add(v1, v1, v2, k_lower, k_upper);
	}
	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	constexpr inline static void add(kAryContainerMeta &v3,
						   const kAryContainerMeta &v1,
						   const kAryContainerMeta &v2,
						   const uint32_t k_lower=0,
						   const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;
		}
	}

	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \param norm = max norm of an dimension which is allowed.
	/// \return true if the element needs to be filtered out. False else.
	constexpr inline static bool add(kAryContainerMeta &v3,
	                                 kAryContainerMeta const &v1,
	                                 kAryContainerMeta const &v2,
						   const uint32_t k_lower,
						   const uint32_t k_upper,
						   const uint32_t norm) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] + v2.__data[i]) % q;
			if ((abs(v3.__data[i]) > norm) && (norm != uint32_t(-1)))
				return true;
		}

		return false;
	}

	/// v1 = v1 - v2, between [k_lower, k_upper)
	/// \param v1 output/input
	/// \param v2 input
	/// \param k_lower lower bound: inclusive
	/// \param k_upper upper bound: exclusive
	constexpr inline static void sub(kAryContainerMeta &v1,
									 const kAryContainerMeta &v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=LENGTH) noexcept {
		sub(v1, v1, v2, k_lower, k_upper);
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \return true if the elements needs to be filled out. False else.
	constexpr inline static void sub(kAryContainerMeta &v3,
	                                 kAryContainerMeta const &v1,
	                                 kAryContainerMeta const &v2,
						   const uint32_t k_lower=0,
						   const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] - v2.__data[i] + q) % q;
		}
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \param norm filter every element out if hte norm is bigger than `norm`
	/// \return true if the elements needs to be filter out. False if not
	constexpr inline static bool sub(kAryContainerMeta &v3,
	                                 kAryContainerMeta const &v1,
	                                 kAryContainerMeta const &v2,
						   			 const uint32_t k_lower,
						   			 const uint32_t k_upper,
						   			 const uint32_t norm) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] - v2.__data[i] + q) % q;
			if ((abs(v3.__data[i]) > norm) && (norm != uint32_t(-1)))
				return true;
		}

		return false;
	}

	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension, inclusive
	/// \param k_upper higher dimension, exclusive
	/// \return v1 == v2 on the coordinates [k_lower, k_higher)
	constexpr inline static bool cmp(kAryContainerMeta const &v1,
	                                 kAryContainerMeta const &v2,
						   const uint32_t k_lower=0,
						   const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

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
	constexpr inline static void set(kAryContainerMeta &v1,
	                                 kAryContainerMeta const &v2,
						   const uint32_t k_lower=0,
						   const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v1.__data[i] = v2.__data[i];
		}
	}

	/// \param obj to compare to
	/// \param k_lower lower coordinate bound, inclusive
	/// \param k_upper higher coordinate bound, exclusive
	/// \return this == obj on the coordinates [k_lower, k_higher)
	constexpr bool is_equal(kAryContainerMeta const &obj,
				  const uint32_t k_lower=0,
				  const uint32_t k_upper=LENGTH) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	/// \return this > obj on the coordinates [k_lower, k_higher)
	constexpr bool is_greater(kAryContainerMeta const &obj,
					const uint32_t k_lower=0,
					const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= LENGTH);
		ASSERT(k_lower < k_upper);

		LOOP_UNROLL();
		for (uint64_t i = k_upper; i > k_lower; i--) {
			if (__data[i - 1] > obj.__data[i - 1])
				return true;
			else if(__data[i - 1] < obj.__data[i - 1])
				return  false;
		}

		return false;
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise, inclusive
	/// \param k_upper higher bound coordinate wise, exclusive
	/// \return this < obj on the coordinates [k_lower, k_higher)
	constexpr bool is_lower(kAryContainerMeta const &obj,
				  const uint32_t k_lower=0,
				  const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= LENGTH);
		ASSERT(k_lower < k_upper);

		LOOP_UNROLL();
		for (uint64_t i = k_upper; i > k_lower; i--) {
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
	constexpr static bool filter(kAryContainerMeta &v3,
	                             const kAryContainerMeta &v1,
	                      		 const kAryContainerMeta &v2,
	                      		 const uint32_t k_lower=0,
	                      		 const uint32_t k_upper=n) {
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
			for(const auto p : {Ts...}) {
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

	/// copy operator
	/// \param obj to copy from
	/// \return this
	kAryContainerMeta & operator =(kAryContainerMeta const &obj) noexcept {
		ASSERT(size() == obj.size() && "äh?");

		if (likely(this != &obj)) { // self-assignment check expected
			__data = obj.__data;
		}

		return *this;
	}

	/// access operator
	/// \param i position. Boundary check is done.
	/// \return limb at position i
	T& operator [](const size_t i) noexcept {
		ASSERT(i < LENGTH);
		return __data[i];
	}
	const T& operator [](const size_t i) const noexcept {
		ASSERT(i < LENGTH);
		return __data[i];
	};

	/// prints this container in binary between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound, inclusive
	/// \param k_upper higher bound, exclusive
	constexpr void print_binary(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < LENGTH && k_upper <= LENGTH && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			unsigned data = (unsigned)__data[i];
			for (int j = 0; j < constexpr_bits_log2(q); ++j) {
				std::cout << (data&1u) << " ";
				data >>= 1;
			}
		}
		std::cout << std::endl;
	}

	/// prints this container between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound, inclusive
	/// \param k_upper higher bound, exclusive
	constexpr void print(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < LENGTH && k_upper <= LENGTH && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << (unsigned)__data[i] << " ";
		}
		std::cout << std::endl;
	}

	/// iterators
	auto begin() noexcept { return __data.begin(); }
	const auto begin() const noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }
	const auto end() const noexcept { return __data.end(); }

	// this data container is never binary
	__FORCEINLINE__ constexpr static bool binary() noexcept { return false; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return LENGTH; }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return LENGTH; }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return LENGTH*sizeof(T); }

	/// returns the underlying data container
	__FORCEINLINE__ constexpr T* ptr() noexcept { return __data.data(); }
	__FORCEINLINE__ constexpr const T* ptr() const noexcept { return __data.data(); }
	__FORCEINLINE__ std::array<T, LENGTH>& data() noexcept { return __data; }
	__FORCEINLINE__ const std::array<T, LENGTH>& data() const noexcept { return __data; }
	constexpr T data(const size_t index) const noexcept { ASSERT(index < LENGTH); return __data[index]; }
	constexpr T get(const size_t index) const noexcept { ASSERT(index < LENGTH); return __data[index]; }
	constexpr void set(const T data, const size_t index) noexcept {
		ASSERT(index < LENGTH);
		__data[index] = data;
	}

	/// sets all elements in the array to the given
	/// \param data
	constexpr void set(const T data) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < n; ++i) {
			__data[i] = data;
		}
	}
protected:
	std::array<T, LENGTH> __data;
};


/// simple data container holding `length` Ts
/// \tparam T base type
/// \tparam length number of elements
template<class T, const uint32_t n, const uint32_t q>
#if __cplusplus > 201709L
	requires kAryContainerAble<T> &&
			 std::is_integral<T>::value
#endif
class kAryContainer_T : public kAryContainerMeta<T, n, q> {
public:
	/// needed constants
	using kAryContainerMeta<T, n, q>::LENGTH;
	using kAryContainerMeta<T, n, q>::MODULUS;

	/// needed typedefs
	using typename kAryContainerMeta<T, n, q>::DataType;
	using typename kAryContainerMeta<T, n, q>::ContainerLimbType;
	using typename kAryContainerMeta<T, n, q>::ContainerType;

	/// needed fields
	using kAryContainerMeta<T, n, q>::__data;

	/// needed functions
	using kAryContainerMeta<T, n, q>::get;
	using kAryContainerMeta<T, n, q>::set;
	using kAryContainerMeta<T, n, q>::neg;
	using kAryContainerMeta<T, n, q>::random;
	using kAryContainerMeta<T, n, q>::zero;
	using kAryContainerMeta<T, n, q>::is_equal;
	using kAryContainerMeta<T, n, q>::is_greater;
	using kAryContainerMeta<T, n, q>::is_lower;
	using kAryContainerMeta<T, n, q>::cmp;
	using kAryContainerMeta<T, n, q>::print;
	using kAryContainerMeta<T, n, q>::size;
	using kAryContainerMeta<T, n, q>::data;
	using kAryContainerMeta<T, n, q>::is_zero;
};

/// NOTE: this implements the representation padded (e.g. every number gets 8 bit)
/// NOTE: this is only the implementation for q=4
template<const uint32_t n>
#if __cplusplus > 201709L
	requires kAryContainerAble<uint8_t>
#endif
class kAryContainer_T<uint8_t, n, 4> : public kAryContainerMeta<uint8_t, n, 4>  {
public:
	/// this is just needed, because Im lazy
	constexpr static uint32_t q = 4;

	/// needed typedefs
	using T = uint8_t;
	using DataType = T;
	using kAryContainerMeta<T, n, q>::LENGTH;
	using kAryContainerMeta<T, n, q>::MODULUS;
	using typename kAryContainerMeta<T, n, q>::ContainerLimbType;
	using typename kAryContainerMeta<T, n, q>::ContainerType;
	using kAryContainerMeta<T, n, q>::__data;

	/// needed functions
	using kAryContainerMeta<T, n, q>::get;
	using kAryContainerMeta<T, n, q>::set;
	using kAryContainerMeta<T, n, q>::neg;
	using kAryContainerMeta<T, n, q>::random;
	using kAryContainerMeta<T, n, q>::zero;
	using kAryContainerMeta<T, n, q>::is_equal;
	using kAryContainerMeta<T, n, q>::is_greater;
	using kAryContainerMeta<T, n, q>::is_lower;
	using kAryContainerMeta<T, n, q>::cmp;
	using kAryContainerMeta<T, n, q>::print;
	using kAryContainerMeta<T, n, q>::size;
	using kAryContainerMeta<T, n, q>::data;
	using kAryContainerMeta<T, n, q>::is_zero;
	using kAryContainerMeta<T, n, q>::add;
	using kAryContainerMeta<T, n, q>::sub;
	using kAryContainerMeta<T, n, q>::ptr;

private:
	// helper sizes
	static constexpr uint32_t limb_u32 	= (n + 	3u)/ 4u;
	static constexpr uint32_t limb_u64 	= (n + 	7u)/ 8u;
	static constexpr uint32_t limb_u128 = (n + 15u)/16u;
	static constexpr uint32_t limb_u256 = (n + 31u)/32u;
	
	// helper masks
	static constexpr __uint128_t mask_4 = (__uint128_t(0x0303030303030303ULL) << 64UL) | (__uint128_t(0x0303030303030303ULL));
	static constexpr __uint128_t mask_q = (__uint128_t(0x0404040404040404ULL) << 64UL) | (__uint128_t(0x0404040404040404ULL));

public:

	/// TODO all these functions should be implemented in the base class too
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
	[[nodiscard]] constexpr static inline T add_T(const T a, const T b) noexcept {
		return (a + b) & mask_4;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a - b
	template<typename T>
	[[nodiscard]] constexpr static inline T sub_T(const T a, const T b) noexcept {
		return (a - b + mask_q) & mask_4;
	}

	///
	/// \tparam T type (probably uint64_t or uint32_t)
	/// \param a
	/// \param b
	/// \return a*b component wise
	template<typename T>
	[[nodiscard]] constexpr static inline T mul_T(const T a, const T b) noexcept {
		constexpr uint32_t nr_limbs = sizeof(T);
		const __uint128_t mask = 0xf;
		__uint128_t c = 0u;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const T a1 = (a >> (8u*i)) & mask;
			const T b1 = (b >> (8u*i)) & mask;
			c ^= (a1*b1) & mask_4;
		}

		/// note implicit call
		return c;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a
	/// \return a%q component wise
	static inline uint8x32_t mod256_T(const uint8x32_t a) noexcept {
		const uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		return uint8x32_t::and_(a, mask256_4);
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a+b, component wise
	static inline uint8x32_t add256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		const uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		return uint8x32_t::and_(uint8x32_t::add(a, b), mask256_4);
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a-b, component wise
	static inline uint8x32_t sub256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		const uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		const uint8x32_t mask256_q = uint8x32_t::set1(0x04);
		return uint8x32_t::and_(uint8x32_t::add(uint8x32_t::sub(a, b), mask256_q), mask256_4);
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a in
	/// \param b in
	/// \return a*b, component wise
	static inline uint8x32_t mul256_T(const uint8x32_t a, const uint8x32_t b) noexcept {
		const uint8x32_t mask256_4 = uint8x32_t::set1(0x03);
		const uint8x32_t tmp = uint8x32_t::mullo(a, b);
		return uint8x32_t::and_(tmp, mask256_4);
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vecot
	static inline void mod(uint8_t *out, const uint8_t *in1) noexcept {
		uint32_t i = 0;
		for (; i < limb_u256; ++i) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + 32*i);
			const uint8x32_t tmp = mod256_T(a);
			uint8x32_t::unaligned_store(out + 32*i, tmp);
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vecot
	static inline void mod(kAryContainer_T &out , const kAryContainer_T &in1) noexcept {
		mod(out.__data.data(), in1.__data.data());
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(uint8_t *out, const uint8_t *in1, const uint8_t *in2) noexcept {
		uint32_t i = 0;
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t b = uint8x32_t::unaligned_load(in2 + i);

			const uint8x32_t tmp = add256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		// NOTE: this is only correct if the inputs where already reduced
		for (; i + 8 < n; i+=8) {
			*((uint64_t *)(out+i)) = add_T<uint64_t>(*((uint64_t *)(in1 + i)), *((uint64_t *)(in2 + i)));
		}

		for (; i < n; i+=1) {
			out[i] = add_T<uint8_t >(in1[i], in2[i]);
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(kAryContainer_T &out,
	                       const kAryContainer_T &in1,
	                       const kAryContainer_T &in2) noexcept {
		add(out.__data.data(), in1.__data.data(), in2.__data.data());
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(uint8_t *out, const uint8_t *in1, const uint8_t *in2) noexcept {
		uint32_t i = 0;
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t b = uint8x32_t::unaligned_load(in2 + i);

			const uint8x32_t tmp = sub256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		for (; i + 8 < n; i+=8) {
			*((uint64_t *)(out+i)) = sub_T<uint64_t>(*((uint64_t *)(in1 + i)), *((uint64_t *)(in2 + i)));
		}

		for (; i < n; i+=1) {
			out[i] = sub_T<uint8_t>(in1[i], in2[i]);
		}
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(kAryContainer_T &out, const kAryContainer_T &in1, const kAryContainer_T &in2) noexcept {
		sub(out.__data.data(), in1.__data.data(), in2.__data.data());
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(uint8_t *out, const uint8_t *in1, const uint8_t *in2) noexcept {
		uint32_t i = 0;
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t b = uint8x32_t::unaligned_load(in2 + i);

			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		//for (; i + 8 < n; i+=8) {
		//	*((uint64_t *)(out+i)) = mul_T<uint64_t>(*((uint64_t *)(in1 + i)), *((uint64_t *)(in2 + i)));
		//}

		for (; i < n; i+=1) {
			out[i] = mul_T<uint8_t>(in1[i], in2[i]);
		}
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(kAryContainer_T &out, const kAryContainer_T &in1, const kAryContainer_T &in2) noexcept {
		mul(out.__data.data(), in1.__data.data(), in2.__data.data());
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(uint8_t *out, const uint8_t *in1, const T in2) noexcept {
		uint32_t i = 0;

		const uint8x32_t b = uint8x32_t::set1(in2);
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		for (; i < n; i+=1) {
			out[i] = mul_T<uint8_t>(in1[i], in2);
		}
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(kAryContainer_T &out, const kAryContainer_T &in1, const T in2) noexcept {
		scalar<T>(out.__data.data(), in1.__data.data(), in2);
	}
};

/// NOTE: this implements the representation padded (e.g. every number gets 8 bit)
/// NOTE: this is only the implementation for q=7
template<const uint32_t n>
#if __cplusplus > 201709L
requires kAryContainerAble<uint8_t>
#endif
class kAryContainer_T<uint8_t, n, 7> : public kAryContainerMeta<uint8_t, n, 7>  {
public:
	/// this is just needed, because Im lazy
	constexpr static uint32_t q = 7;

	/// needed typedefs
	using T = uint8_t;
	using DataType = T;
	using kAryContainerMeta<T, n, q>::LENGTH;
	using kAryContainerMeta<T, n, q>::MODULUS;
	using typename kAryContainerMeta<T, n, q>::ContainerLimbType;
	using typename kAryContainerMeta<T, n, q>::ContainerType;
	using kAryContainerMeta<T, n, q>::__data;

	/// needed functions
	using kAryContainerMeta<T, n, q>::get;
	using kAryContainerMeta<T, n, q>::set;
	using kAryContainerMeta<T, n, q>::neg;
	using kAryContainerMeta<T, n, q>::random;
	using kAryContainerMeta<T, n, q>::zero;
	using kAryContainerMeta<T, n, q>::is_equal;
	using kAryContainerMeta<T, n, q>::is_greater;
	using kAryContainerMeta<T, n, q>::is_lower;
	using kAryContainerMeta<T, n, q>::cmp;
	using kAryContainerMeta<T, n, q>::print;
	using kAryContainerMeta<T, n, q>::size;
	using kAryContainerMeta<T, n, q>::data;
	using kAryContainerMeta<T, n, q>::is_zero;
	using kAryContainerMeta<T, n, q>::add;
	using kAryContainerMeta<T, n, q>::sub;
	using kAryContainerMeta<T, n, q>::ptr;

private:
	// helper sizes
	static constexpr uint32_t limb_u32 	= (n + 	3u)/ 4u;
	static constexpr uint32_t limb_u64 	= (n + 	7u)/ 8u;
	static constexpr uint32_t limb_u128 = (n + 15u)/16u;
	static constexpr uint32_t limb_u256 = (n + 31u)/32u;

	// helper masks
	/// TODO remove
	static const uint64_t mask_4 = 4;
	static const uint64_t mask_q = 4;
	static constexpr __uint128_t mask_7 = (__uint128_t(0x0707070707070707ULL) << 64UL) | (__uint128_t(0x0707070707070707ULL));
	static constexpr __uint128_t mask_f = (__uint128_t(0x0f0f0f0f0f0f0f0fULL) << 64UL) | (__uint128_t(0x0f0f0f0f0f0f0f0fULL));
	static constexpr __uint128_t mask_1 = (__uint128_t(0x0101010101010101ULL) << 64UL) | (__uint128_t(0x0101010101010101ULL));
	static constexpr __uint128_t mask_3 = (__uint128_t(0x0303030303030303ULL) << 64UL) | (__uint128_t(0x0303030303030303ULL));

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
			const T a1 = (a >> (8u*i)) & mask;
			const T b1 = (b >> (8u*i)) & mask;
			c ^= mod_T(a1*b1) << (8u*i);
		}

		return c;
	}

	/// helper function
	/// vectorized version, input are 32x 8Bit vectors
	/// \param a
	/// \return a%q component wise
	static inline uint8x32_t mod256_T(const uint8x32_t aa) noexcept {
		const uint8x32_t mask256_7 = uint8x32_t::set1(0x07);
		const uint8x32_t mask256_1 = uint8x32_t::set1(0x01);
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

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vecot
	static inline void mod(uint8_t *out, const uint8_t *in1) noexcept {
		uint32_t i = 0;
		/// TODO loop unrolling, tail managment
		for (; i < limb_u256; ++i) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + 32*i);
			const uint8x32_t tmp = mod256_T(a);
			uint8x32_t::unaligned_store(out + 32*i, tmp);
		}
	}

	/// NOTE: inplace
	/// computes mod q
	/// \param out = in1 % q
	/// \param in1: input vecot
	static inline void mod(kAryContainer_T &out , const kAryContainer_T &in1) noexcept {
		mod(out.__data.data(), in1.__data.data());
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(uint8_t *out, const uint8_t *in1, const uint8_t *in2) noexcept {
		uint32_t i = 0;
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t b = uint8x32_t::unaligned_load(in2 + i);

			const uint8x32_t tmp = add256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		// NOTE: this is only correct if the inputs where already reduced
		for (; i + 8 < n; i+=8) {
			*((uint64_t *)(out+i)) = add_T<uint64_t>(*((uint64_t *)(in1 + i)), *((uint64_t *)(in2 + i)));
		}

		for (; i < n; i+=1) {
			out[i] = add_T<uint8_t >(in1[i], in2[i]);
		}
	}

	/// vector addition
	/// \param out = in1 + in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void add(kAryContainer_T &out, const kAryContainer_T &in1, const kAryContainer_T &in2) noexcept {
		add(out.__data.data(), in1.__data.data(), in2.__data.data());
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(uint8_t *out, const uint8_t *in1, const uint8_t *in2) noexcept {
		uint32_t i = 0;
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t b = uint8x32_t::unaligned_load(in2 + i);

			const uint8x32_t tmp = sub256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		for (; i + 8 < n; i+=8) {
			*((uint64_t *)(out+i)) = sub_T<uint64_t>(*((uint64_t *)(in1 + i)), *((uint64_t *)(in2 + i)));
		}

		for (; i < n; i+=1) {
			out[i] = sub_T<uint8_t>(in1[i], in2[i]);
		}
	}

	/// vector subtract
	/// \param out = in1 - in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void sub(kAryContainer_T &out, const kAryContainer_T &in1, const kAryContainer_T &in2) noexcept {
		sub(out.__data.data(), in1.__data.data(), in2.__data.data());
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(uint8_t *out, const uint8_t *in1, const uint8_t *in2) noexcept {
		uint32_t i = 0;
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t b = uint8x32_t::unaligned_load(in2 + i);

			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		//for (; i + 8 < n; i+=8) {
		//	*((uint64_t *)(out+i)) = mul_T<uint64_t>(*((uint64_t *)(in1 + i)), *((uint64_t *)(in2 + i)));
		//}

		for (; i < n; i+=1) {
			out[i] = mul_T<uint8_t>(in1[i], in2[i]);
		}
	}

	/// components-wise vector multiplication
	/// \param out = in1*in2
	/// \param in1 input: vector
	/// \param in2 input: vector
	static inline void mul(kAryContainer_T &out, const kAryContainer_T &in1, const kAryContainer_T &in2) noexcept {
		mul(out.__data.data(), in1.__data.data(), in2.__data.data());
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(uint8_t *out, const uint8_t *in1, const T in2) noexcept {
		uint32_t i = 0;

		const uint8x32_t b = uint8x32_t::set1(in2);
		for (; i+32 < n; i+=32) {
			const uint8x32_t a = uint8x32_t::unaligned_load(in1 + i);
			const uint8x32_t tmp = mul256_T(a, b);
			uint8x32_t::unaligned_store(out + i, tmp);
		}

		for (; i < n; i+=1) {
			out[i] = mul_T<uint8_t>(in1[i], in2);
		}
	}

	/// scalar multiplication
	/// \param out = in1*in2
	/// \param in1 output: vector
	/// \param in2 output: scalar
	template<typename T>
	static inline void scalar(kAryContainer_T &out, const kAryContainer_T &in1, const T in2) noexcept {
		scalar<T>(out.__data.data(), in1.__data.data(), in2);
	}
};
#endif
