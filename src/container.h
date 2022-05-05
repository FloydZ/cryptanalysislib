#ifndef SMALLSECRETLWE_CONTAINER_H
#define SMALLSECRETLWE_CONTAINER_H

#include <array>
#include <cstdint>
#include <atomic>

#if defined(USE_AVX) || defined(USE_AVX2)
#include <immintrin.h>
#include <emmintrin.h>
#endif

// local includes
#include "random.h"
#include "helper.h"

// external includes
#include "m4ri/m4ri.h"

// TODO C++ 20 comparsion operator
// C macro for implementing multi limb comparison.
#define BINARYCONTAINER_COMPARE(limb1, limb2, op1, op2) \
if (limb1 op1 limb2)                                    \
	return 1;                                           \
else if(limb1 op2 limb2)                                \
	return 0;

// C Macro for implementing multi limb comparison
#define BINARYCONTAINER_COMPARE_MASKED(limb1, limb2, mask, op1, op2)\
if ((limb1&mask) op1 (limb2&mask))                                  \
	return 1;                                                       \
else if((limb1&mask) op2 (limb2&mask))                              \
	return 0;                                                       \

#ifdef USE_AVX2
static inline uint32_t hammingweight_mod2_limb256(__m256i v) {
	const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
	                                        2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
	                                        1, 2, 2, 3, 2, 3, 3, 4);
	const __m256i low_mask = _mm256_set1_epi8(0x0f);
	const __m256i lo =_mm256_and_si256(v,low_mask);
	const __m256i hi = _mm256_and_si256( _mm256_srli_epi32(v, 4), low_mask);
	const __m256i popcnt1 = _mm256_shuffle_epi8(lookup,lo);
	const __m256i popcnt2 = _mm256_shuffle_epi8(lookup,hi);
	const __m256i total = _mm256_add_epi8(popcnt1,popcnt2);

	const __m256i final = _mm256_sad_epu8(total, _mm256_setzero_si256());

	// propably not fast
	alignas(32) static uint64_t vec[4];
	_mm256_store_si256((__m256i *)vec , final);
	return vec[0] + vec[1] + vec[2] + vec[3];

	//const __m256i final2 = _mm256_hadd_epi32(final, final);
	//return _mm256_cvtsi256_si32(final2);

	//const __m256i values = _mm256_hadd_epi32(final, _mm256_permute2x128_si256(final, final, 1));
	//const __m256i values2 = _mm256_hadd_epi32(values, values);
	//return _mm256_cvtsi256_si32(values2);
}
static inline __m256i hammingweight_mod2_limb256_nonacc(__m256i v) {
	const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
	                                        2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
	                                        1, 2, 2, 3, 2, 3, 3, 4);
	const __m256i low_mask = _mm256_set1_epi8(0x0f);
	const __m256i lo = _mm256_and_si256(v, low_mask);
	const __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
	const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
	const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
	const __m256i total = _mm256_add_epi8(popcnt1, popcnt2);

	const __m256i final = _mm256_sad_epu8(total, _mm256_setzero_si256());
	return final;
}
#endif

/// Concept for base data type.
/// \tparam T
template<typename T>
concept kAryContainerAble =
requires(T t) {
	t ^ t;
	t.random();
	T(0);
	//~(T(0));
};

/// Concept for the base data type
/// \tparam T
template<typename T>
concept kAryPackedContainerAble =
std::is_integral<T>::value && requires(T t) {
	t ^ t;
	t + t;
};

/// Concept fot the base data type
/// \tparam T
template<typename T>
concept BinaryContainerAble =
std::is_integral<T>::value && requires(T t) {
	t ^ t;
	t & t;
	t | t;
};

/// simple data container holding `length` Ts
/// \tparam T base type
/// \tparam length number of elements
template<class T, uint32_t length>
	requires kAryContainerAble<T>
class kAryContainer_T {
public:

	// Needed for the internal template system.
	typedef T DataType;
	typedef T ContainerLimbType;

	// internal data length. Used in the template system to pass through this information
	constexpr static uint32_t LENGTH = length;

	/// zeros our the whole container
	/// \return nothing
	constexpr inline void zero() noexcept {
		LOOP_UNROLL();
		for (unsigned int i = 0; i < length; i++){
			__data[i] = T(0);
		}
	}

	/// set everything on `fff.fff`
	/// \return nothing
	constexpr inline void one() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < length; i++) {
			__data[i] = ~(T(0));
		}
	}

	/// generates random coordinates
	/// \param k_lower lower coordinate to start from
	/// \param k_higher higher coordinate to stop. Not included.
	void random(uint32_t k_lower=0, uint32_t k_higher=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_higher; i++){
			__data[i].random();
		}
	}

	/// checks if every dimension is zero
	/// \return true/false
	[[nodiscard]] bool is_zero() const noexcept {
		for (uint32_t i = 0; i < length; ++i) {
			if(__data[i] != T(0))
				return false;
		}

		return true;
	}

	/// calculate the hamming weight
	/// \return the hamming weight
	inline uint32_t weight() const noexcept {
		uint32_t r = 0;
		for (int i = 0; i < length; ++i) {
			if (__data[i] != 0)
				r += 1;
		}
		return r;
	}

	/// swap coordinate i, j, boundary checks are done
	/// \param i coordinate
	/// \param j coordinate
	void swap(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < length && j < length);
		SWAP(__data[i], __data[j]);
	}

	/// *-1
	/// \param i
	void flip(const uint32_t i) noexcept {
		ASSERT(i < length);
		__data[i] *= -1;
	}

	/// negate every coordinate between [k_lower, k_higher)
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	inline void neg(const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length &&k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			__data[i] = 0 - __data[i];
		}
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	inline static void add(kAryContainer_T &v3, kAryContainer_T const &v1, kAryContainer_T const &v2,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = v1.__data[i] + v2.__data[i];
		}
	}

	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \param norm = max norm of an dimension which is allowed.
	/// \return true if the element needs to be filtered out. False else.
	inline static bool add(kAryContainer_T &v3, kAryContainer_T const &v1, kAryContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper,
	                       const uint32_t norm) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = v1.__data[i] + v2.__data[i];
			if ((abs(v3.__data[i]) > norm) && (norm != uint32_t(-1)))
				return true;
		}

		return false;
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \return true if the elements needs to be filled out. False else.
	inline static void sub(kAryContainer_T &v3, kAryContainer_T const &v1, kAryContainer_T const &v2,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = v1.__data[i] - v2.__data[i];
		}
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \param norm filter every element out if hte norm is bigger than `norm`
	/// \return true if the elements needs to be filter out. False if not
	inline static bool sub(kAryContainer_T &v3, kAryContainer_T const &v1, kAryContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper,
	                       const uint32_t norm) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = v1.__data[i] - v2.__data[i];
			if ((abs(v3.__data[i]) > norm) && (norm != uint32_t(-1)))
				return true;
		}

		return false;
	}

	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \return v1 == v2 on the coordinates [k_lower, k_higher)
	inline static bool cmp(kAryContainer_T const &v1, kAryContainer_T const &v2,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		return true;
	}

	/// static function. Sets v1 to v2 between [k_lower, k_higher). Does not touch the other coordinates in v1
	/// \param v1 output container
	/// \param v2 input container
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	inline static void set(kAryContainer_T &v1, kAryContainer_T const &v2,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v1.__data[i] = v2.__data[i];
		}
	}

	/// \param obj to compare to
	/// \param k_lower lower coordinate bound
	/// \param k_upper higher coordinate bound
	/// \return this == obj on the coordinates [k_lower, k_higher)
	bool is_equal(kAryContainer_T const &obj,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		return cmp(this, obj, k_lower, k_upper);
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	/// \return this > obj on the coordinates [k_lower, k_higher)
	bool is_greater(kAryContainer_T const &obj,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= length && "ERROR is_greater not correct k_upper");
		ASSERT(k_lower < k_upper && "ERROR is_greater not correct k_lower");

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
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	/// \return this < obj on the coordinates [k_lower, k_higher)
	bool is_lower(kAryContainer_T const &obj,
	                     const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= length && "ERROR is_lower not correct k_upper");
		ASSERT(k_lower < k_upper && "ERROR is_lower not correct k_lower");

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

	/// access operator
	/// \param i position. Boundary check is done.
	/// \return limb at position i
	T& operator [](size_t i) noexcept {
		ASSERT(i < length && "wrong access index");
		return __data[i];
	}
	const T& operator [](const size_t i) const noexcept {
		ASSERT(i < length && "wrong access index");
		return __data[i];
	};

	/// copy operator
	/// \param obj to copy from
	/// \return this
	kAryContainer_T& operator =(kAryContainer_T const &obj) noexcept {
		ASSERT(size() == obj.size() && "äh?");

		if (likely(this != &obj)) { // self-assignment check expected
			__data = obj.__data;
		}

		return *this;
	}

	/// prints this container between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound
	/// \param k_upper higher bound
	void print(const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_lower < length && k_upper < length && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << __data[i] << " ";
		}
		std::cout << "\n";
	}

	/// iterators
	auto begin() noexcept { return __data.begin();}
	const auto begin() const noexcept { return __data.begin();}
	auto end() noexcept { return __data.end();}
	const auto end() const noexcept { return __data.end();}

	// this data container is never binary
	__FORCEINLINE__ constexpr static bool binary() noexcept { return false; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return length; }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return length; }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return length*sizeof(T); }

	/// returns the underlying data container
	__FORCEINLINE__ std::array<T, length>& data() noexcept { return __data; }
	__FORCEINLINE__ const std::array<T, length>& data() const noexcept { return __data; }
	//T& data(const uint64_t index) { ASSERT(index < length && "wrong index"); return __data[index]; }
	const T data(const uint64_t index) const noexcept { ASSERT(index < length && "wrong index"); return __data[index]; }

	/// TODO remove
	T get_type() noexcept {return __data[0]; }
private:
	std::array<T, length> __data;
};

/// represents a vector of numbers mod `MOD` in vector of `T` in a compressed way
template<class T, const T MOD, uint32_t length>
	requires kAryPackedContainerAble<T>
class kAryPackedContainer_T {
	/// Nomenclature:
	///     Number 	:= actual data one wants to save % MOD
	///		Limb 	:= Underlying data container holding at max sizeof(T)/log2(MOD) many numbers.
	/// Its not possible, that numbers cover more than one limb.
	/// The internal data container layout looks like this:
	/// 		limb0				limb1			limb2
	///   [	n0	,  n1  ,  n2  |	    , 	,	  |		,	 ,     |  .... ]
	/// The container fits as much numbers is the limb as possible. But there will no overhanging elements (e.g.
	///  numbers that first bits are on one limb and the remaining bits are on the next limb).
public:
	// base constructor
	kAryPackedContainer_T () { __data.fill(0); }

	// number of bits in each T
	constexpr static uint16_t bits_per_limb = sizeof(T)*8;
	// number of bits needed to represent MOD
	constexpr static uint16_t bits_per_number = (uint16_t)const_log(MOD) + 1;
	// number of numbers one can fit into each limb
	constexpr static uint16_t numbers_per_limb = (sizeof(T)*8u) / bits_per_number;
	// Number of Limbs needed to represent `length` numbers of size log(MOD) +1
	constexpr static uint16_t internal_limbs = MAX(1, (length+numbers_per_limb-1)/ numbers_per_limb);
	// mask with the first `bits_per_number` set to one
	constexpr static T number_mask = (T(1) << (bits_per_number)) - 1;

	// true if we need every bit of the last bit
	constexpr static bool is_full = (length%bits_per_limb) == 0;

	// minimal internal datatype to present an element.
	using DataType = LogTypeTemplate<bits_per_number>;

	// we are good C++ devs.
	typedef T ContainerLimbType;

	// internal data
	std::array<T, internal_limbs> __data;

	// list compatibility typedef
	typedef kAryPackedContainer_T<T, MOD, length> ContainerType;
	typedef T LimbType;
	typedef T LabelContainerType;

	// make the length of the container public available
	constexpr static uint32_t LENGTH = length;

	/// the mask is only valid for one internal number.
	/// \param i bit position the read
	/// \return bit mask to access the i-th element within a limb
	inline T accessMask(const uint16_t i) const noexcept {
		return number_mask << (i%numbers_per_limb);
	}

	/// access the i-th coordinate/number/element
	/// \param i coordinate to access.
	/// \return the number you wanted to access, shifted down to the lowest bits.
	DataType get(const uint16_t i) const noexcept {
		// needs 5 instructions. So 64*5 for the whole limb
		ASSERT(i < length);
		return DataType((__data[i/numbers_per_limb] >> ((i%numbers_per_limb)*bits_per_number)) & number_mask);
	}

	/// sets the `i`-th number to `data`
	/// \param data
	/// \param i -th number to overwrite
	/// \retutn nothing
	void set(const DataType data, const uint16_t i) noexcept {
		ASSERT(i < length);
		const uint16_t off = i/numbers_per_limb;
		const uint16_t spot = (i%numbers_per_limb) * bits_per_number;

		T bla = (number_mask&T(data%MOD)) << spot;
		T val = __data[off] & (~(number_mask << spot));
		__data[off] = bla | val;
	}

	/// set everything to zero
	constexpr void zero() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < internal_limbs; i++){
			__data[i] = 0;
		}
	}

	/// sets everything to zero between [a, b)
	/// \param a lower bound
	/// \param b higher bound
	/// \return nothing
	constexpr void zero(const uint32_t a, const uint32_t b) noexcept {
		for (uint32_t i = a; i < b; i++){
			set(0, i);
		}
	}

	/// set everything to one
	/// \return nothing
	constexpr void one(const uint32_t a=0, const uint32_t b=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(1, i);
		}
	}

	/// Set everything to two
	/// \return nothing
	constexpr void two(const uint32_t a=0, const uint32_t b=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(2%MOD, i);
		}
	}

	/// generates all limbs uniformly random
	/// \return nothing
	void random(const uint32_t a=0, const uint32_t b=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i=a; i < b; i++) {
			set(fastrandombytes_uint64() % MOD, i);
		}
	}

	/// return the positions of the first p bits/numbers set
	/// \param out output: array of the first p positions set in the container
	/// \param p maximum bits!=0 to find
	void get_bits_set(uint16_t *out, const uint32_t p) const noexcept {
		uint32_t ctr = 0;
		for (uint32_t i = 0; i < LENGTH; i++) {
  			if (unsigned(get(i)) != 0u) {
				out[ctr] = i;
				ctr += 1;
			}

			// early exit
			if (ctr == p) {
				return;
			}
  		}
	}

	/// \return true if every limb is empty
	bool is_zero() const noexcept {
		for (uint32_t i = 0; i < internal_limbs; ++i) {
			if(__data[i] != 0)
				return false;
		}

		return true;
	}

	/// checks if every number between [a, b) is empyt
	/// \param a lower bound
	/// \param b higher bound
	/// \return true/false
	bool is_zero(const uint32_t a, const uint32_t b) const noexcept {
		for (uint32_t i = a; i < b; ++i) {
			if(get(i) != 0)
				return false;
		}

		return true;
	}

	/// calculates the hamming weight between [a,b)
	/// \param a lower bound
	/// \param b higher bound
	/// \return the hamming weight
	inline uint32_t weight(const uint32_t a=0, const uint32_t b=LENGTH) const noexcept {
		uint64_t r = 0;
		for (uint32_t i = a; i < b; ++i) {
			if (get(i) != 0)
				r += 1;
		}

		return r;
	}

	/// swap the numbers on positions `i`, `j`.
	/// \param i first coordinate
	/// \param j second coordinate
	void swap(const uint16_t i, const uint16_t j) {
		ASSERT(i < length && j < length);
		auto tmp = get(i);
		set(i, get(j));
		set(j, tmp);
	}

	/// TODO generalize to arbitrary MOD
	inline void neg() noexcept {
		if constexpr (internal_limbs == 1) {
			__data[0] = neg_mod3_limb(__data[0]);
			return;
		}

		uint32_t i = 0;
//#ifdef USE_AVX2
//		for (; i+4 <= internal_limbs; i += 4){
//			__m256i t = neg_mod3_limb256(*((__m256i *)&__data[i]));
//			*((__m256i *)&__data[i]) = t;
//		}
//#endif
		for (; i+2 <= internal_limbs; i += 2) {
			__uint128_t t = neg_mod3_limb128(*((__uint128_t *)&__data[i]));
			*((__uint128_t *)&__data[i]) = t;
		}

		for (; i < internal_limbs; i++) {
			__data[i] = neg_mod3_limb(__data[i]);
		}

	}

	constexpr inline void neg(const uint32_t k_lower, const uint32_t k_higher) {
		// TODO
	}

	/// infix negates (x= -x mod q)  all numbers between [k_lower, k_higher)
	/// \param k_lower lower limit
	/// \param k_upper higher limit
	inline void neg_slow(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= length &&k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			set(((-get(i)) + MOD)%MOD, i);
		}
	}


	/// calcs the hamming weight of one limb.
	/// NOTE only correct if there is no 3 in one of the limbs
	static inline uint16_t hammingweight_mod3_limb(const T a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr T c1 = T(6148914691236517205u);
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T c2 = T(12297829382473034410u);

		const T ac1 = a&c1; // filter the ones
		const T ac2 = a&c2; // filter the twos

		return __builtin_popcountll(ac1) + __builtin_popcountll(ac2);
	}

	/// calcs the hamming weight of one __uint128
	/// NOTE only correct if there is no 3 in one of the limbs
	static inline uint16_t hammingweight_mod3_limb128(const __uint128_t a) noexcept {
		constexpr __uint128_t c1 = __uint128_t(6148914691236517205u)   << 64 | 6148914691236517205u;
		constexpr __uint128_t c2 = __uint128_t(12297829382473034410u)  << 64 | 12297829382473034410u;

		const __uint128_t ac1 = a&c1; // filter the ones
		const __uint128_t ac2 = a&c2; // filter the twos

		// can this thing get faster?
		return __builtin_popcountll(ac1) + __builtin_popcountll(ac1>>64) +
			   __builtin_popcountll(ac2) + __builtin_popcountll(ac2>>64);
	}

#ifdef USE_AVX2
	static inline uint16_t hammingweight_mod3_limb256(const __m256i a) noexcept {
		constexpr __uint128_t c1_128 = __uint128_t(6148914691236517205u)   << 64 | 6148914691236517205u;
		constexpr __uint128_t c2_128 = __uint128_t(12297829382473034410u)  << 64 | 12297829382473034410u;
		const static __m256i c1 = _mm256_set_m128i((__m128i)c1_128, (__m128i)c1_128);
		const static __m256i c2 = _mm256_set_m128i((__m128i)c2_128, (__m128i)c2_128);

		const __m256i ac1 = _mm256_and_si256(a, c1); // filter the ones
		const __m256i ac2 = _mm256_and_si256(a, c2); // filter the twos

		return hammingweight_mod2_limb256(ac1) + hammingweight_mod2_limb256(ac2);
	}
#endif

	/// TODO only valid for ternary
	/// \tparam k_lower lower limit
	/// \tparam k_upper
	template<uint32_t k_lower, uint32_t k_upper>
	inline void neg() noexcept {
		static_assert(k_upper <= length && k_lower < k_upper);

		constexpr uint32_t ll = 2*k_lower/bits_per_limb;
		constexpr uint32_t lh = 2*k_upper/bits_per_limb;
		constexpr uint32_t ol = 2*k_lower%bits_per_limb;
		constexpr uint32_t oh = 2*k_upper%bits_per_limb;
		constexpr T ml = ~((T(1) << ol) - T(1));
		constexpr T mh = (T(1) << oh) - T(1);

		constexpr T nml = ~ml;
		constexpr T nmh = ~mh;

		if constexpr (ll == lh) {
			constexpr T m = ml&mh;
			constexpr T nm = ~m;
			__data[ll] = (neg_mod3_limb(__data[ll])&m)^(__data[ll]&nm);
		} else {
			for (uint32_t i = ll+1; i < lh-1; ++i) {
				__data[i] = neg_mod3_limb(__data[i]);
			}
			__data[ll] = (neg_mod3_limb(__data[ll])&ml)^(__data[ll]&nml);
			__data[lh] = (neg_mod3_limb(__data[lh])&mh)^(__data[lh]&nmh);
		}

	}

	/// negates one limb
	/// \param a input limb
	/// \return negative limb
	static inline T neg_mod3_limb(const T a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr T c1 = T(6148914691236517205u);
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T c2 = T(12297829382473034410u);

		const T e1 = a&c1; // filter the ones
		const T e2 = a&c2; // filter the twos

		// reshifts everything to the correct place
		return (e1 << 1) ^ (e2 >> 1);
	}

	///
	/// \param a
	/// \return
	static inline __uint128_t neg_mod3_limb128(const __uint128_t a) noexcept {
		constexpr __uint128_t c2 = __uint128_t(12297829382473034410u)  << 64 | 12297829382473034410u;
		constexpr __uint128_t c1 = __uint128_t(6148914691236517205u)   << 64 | 6148914691236517205u;

		const __uint128_t e1 = a&c1;
		const __uint128_t e2 = a&c2;
		return (e1 << 1) ^ (e2 >> 1);
	}

#ifdef USE_AVX2
	static inline __m256i neg_mod3_limb256(const __m256i a) noexcept {
		const static __m256i c1 = _mm256_set_epi64x(6148914691236517205u,6148914691236517205u,6148914691236517205u,6148914691236517205u);
		const static __m256i c2 = _mm256_set_epi64x(12297829382473034410u,12297829382473034410u,12297829382473034410u,12297829382473034410u);

		const __m256i e1 = _mm256_and_si256(a,c1);
		const __m256i e2 = _mm256_and_si256(a,c2);

		const __m256i e11 = _mm256_slli_epi64(e1, 1);
		const __m256i e21 = _mm256_srli_epi64(e2, 1);

		return _mm256_xor_si256(e11, e21);
	}
#endif

	/// calculates a mod 3 on everything number
	/// \param a
	/// \return
	static inline T mod3_limb(const T a) noexcept {
		const T e = mod3_limb_withoutcorrection(a);

		// fix the overflow on coordinate 32
		constexpr T ofmask = (T(1u) << 62) - 1;
		const T ofbit = ((a>>62)%3) << 62;
		return (e&ofmask)^ofbit;
	}

	// same as mod3_limb but without the correction of the last entry
	static inline T mod3_limb_withoutcorrection(const T a) noexcept {
		// int(0b1100110011001100110011001100110011001100110011001100110011001100)
		constexpr T f = T(14757395258967641292u);
		// int(0b001100110011001100110011001100100110011001100110011001100110011)
		constexpr T g = T(3689348814741910323u);
		// int(0b0100010001000100010001000100010001000100010001000100010001000100)
		constexpr T c1 = T(4919131752989213764u);
		// int(0b0001000100010001000100010001000100010001000100010001000100010001)
		constexpr T c2 = T(1229782938247303441u);
		const T c = a&f;
		const T d = a&g;

		const T cc = ((c+c1) >> 2)&f; // adding one to simulate the carry bit
		const T dc = ((d+c2) >> 2)&g;

		const T cc2 = c + cc;
		const T dc2 = d + dc;

		const T cf = cc2&f;     // filter out again resulting carry bits
		const T dg = dc2&g;
		const T e = (cf^dg);

		return e;
	}

	static inline __uint128_t mod3_limb128(const __uint128_t a) noexcept {
		const __uint128_t e = mod3_limb_withoutcorrection(a);

		// fix the overflow on
		constexpr __uint128_t ofmask2 = ((__uint128_t(1u) << 126u) - 1u) ^ __uint128_t(~((T(1u)<<62u)-1u));
		constexpr __uint128_t ofmaskh = (T(1u) << 2u) - 1u;
		constexpr __uint128_t ofmask  = (__uint128_t(ofmaskh) << 64u) ^ __uint128_t(ofmaskh);
		constexpr __uint128_t ofcarry = (__uint128_t(1u) << 64u) + 1u;

		const __uint128_t ofbits  = (a>>62)&ofmask;
		const __uint128_t ofbitsc = ((ofbits + ofcarry) >> 2) & ofmask;
		const __uint128_t ofbits2 = (ofbits + ofbitsc) & ofmask;
		const __uint128_t ofbits3 = ofbits2 << 62;
		return (e&ofmask2)^ofbits3;
	}


	/// \param x input number
	/// \param y output number
	/// \return x-y in every number
	static inline T sub_mod3_limb(const T x, const T y) noexcept {
		return add_mod3_limb(x, neg_mod3_limb(y));
	}

	///
	/// \param x
	/// \param y
	/// \return
	static inline __uint128_t sub_mod3_limb128(const __uint128_t x, const __uint128_t y) noexcept {
		return add_mod3_limb128(x, neg_mod3_limb128(y));
	}

#ifdef USE_AVX2
	///
	/// \param x
	/// \param y
	/// \return
	static inline __m256i sub_mod3_limb256(const __m256i x, const __m256i y) noexcept {
		return add_mod3_limb256(x, neg_mod3_limb256(y));
	}
#endif

	template<typename TT>
	static inline TT add_mod3_limb_no_overflow(const TT a, const TT b) noexcept {
		// we know that no overflow will happen, So no 2+2=1
		return a+b;
	}

	///
	/// \param x input first number
	/// \param y input second number
	/// \return x+y in every number
	static inline T add_mod3_limb(const T x, const T y) noexcept {
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T c2 = T(12297829382473034410u);
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr T c1 = T(6148914691236517205u);

		// These are not the optimal operations to calculate the ternary addition. But nearly.
		// The problem is that one needs to spit the limb for the ones/two onto two separate limbs. But two separate
		// limbs mean
		//      - higher memory consumption for each container
		//      - complicated hashing for the hashmaps
		const T xy = x^y;
		const T xy2 = x&y;
		const T a = xy&c1;
		const T b = xy&c2;
		const T c = xy2&c1;
		const T d = xy2&c2;
		const T e = a & (b>>1);

		const T r0 = e ^ (d >> 1) ^ (a);
		const T r1 = (e << 1) ^ (b) ^ ( c << 1);
		return r0 ^ r1;
	}


	static inline __uint128_t add_mod3_limb128(const __uint128_t x, const __uint128_t y) noexcept {
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr __uint128_t c2  = __uint128_t(12297829382473034410u) << 64 | 12297829382473034410u;
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr __uint128_t c1  = __uint128_t(6148914691236517205u) << 64 | 6148914691236517205u;

		const __uint128_t xy = x^y;
		const __uint128_t xy2 = x&y;
		const __uint128_t a = xy&c1;
		const __uint128_t b = xy&c2;
		const __uint128_t c = xy2&c1;
		const __uint128_t d = xy2&c2;
		const __uint128_t e = a & (b>>1);

		const __uint128_t r0 = e ^ (d >> 1) ^ (a);
		const __uint128_t r1 = (e << 1) ^ (b) ^ ( c << 1);
		return r0 ^ r1;

	}

#ifdef USE_AVX2
	static inline __m256i add_mod3_limb_no_overflow(const __m256i a, const __m256i b) noexcept {
		// no overflow will happen.
		return _mm256_add_epi64(a,b);
	}


	static inline __m256i add_mod3_limb256(const __m256i x, const __m256i y) noexcept {
		const static __m256i c1 = _mm256_set_epi64x(6148914691236517205u,6148914691236517205u,6148914691236517205u,6148914691236517205u);
		const static __m256i c2 = _mm256_set_epi64x(12297829382473034410u,12297829382473034410u,12297829382473034410u,12297829382473034410u);

		const __m256i xy = _mm256_xor_si256(x,y);
		const __m256i xy2 = _mm256_and_si256(x,y);
		const __m256i a = _mm256_and_si256(xy,c1);
		const __m256i b = _mm256_and_si256(xy,c2);
		const __m256i c = _mm256_and_si256(xy2,c1);
		const __m256i d = _mm256_and_si256(xy2,c2);

		const __m256i e = _mm256_and_si256(a, _mm256_srli_epi64(b, 1));

		const __m256i r0 = _mm256_xor_si256(e, _mm256_xor_si256(a, _mm256_srli_epi64(d, 1)));
		const __m256i r1 = _mm256_xor_si256(b, _mm256_xor_si256(_mm256_slli_epi64(e, 1), _mm256_slli_epi64(c, 1)));

		return _mm256_xor_si256(r0, r1);
	}
#endif

	// returns 2*a, input must be reduced mod 3
	static inline T times2_mod3_limb(const T a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr T t1 = T(6148914691236517205);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T t2 = T(12297829382473034410u);

		const T tm1 = a&t1; // extract the ones
		const T tm2 = a&t2; // extract the twos
		const T acc = ((tm1<<1)^tm2); // where are not zeros
		const T b = add_mod3_limb(a, t2&acc); // add two
		return b^(tm1<<1);
	}

	/// \param a element to check.
	/// \return true if `a` contains a two at any coordinate
	static inline bool filter2_mod3_limb(const T a) noexcept {
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		return (a&m) != 0;
	}

	/// \param a element to check
	/// \param limit
	/// \return returns the number of two in the limb
	static inline uint32_t filter2count_mod3_limb(const T a) noexcept {
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		return __builtin_popcountll(a&m);
	}

	static inline uint32_t filter2count_mod3_limb128(const __uint128_t a) noexcept {
		constexpr __uint128_t m = __uint128_t(12297829382473034410u) << 64 ^ 12297829382473034410u;
		const __uint128_t am = a&m;
		return __builtin_popcountll(am>>64) + __builtin_popcountll(am);
	}

	/// \tparam k_lower lower limit in coordinates to check if twos exist
	/// \tparam k_upper upper limit in coordinates (not bits)
	/// \param a element to check if two exists
	/// \param limit how many twos are in total allowed
	/// \return return the twos in a[k_lower, k_upper].
	template<const uint16_t k_lower, const uint16_t k_upper>
	static inline uint32_t filter2count_range_mod3_limb(const T a) noexcept {
		static_assert(k_lower != 0 && k_lower < k_upper && k_upper <= length);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		constexpr T mask = ((T(1u) << (2u*k_lower)) - 1u) & ((T(1u) << (2u*k_upper)) - 1u);
		return __builtin_popcountll(a&mask&m);
	}

	template<const uint16_t k_upper>
	inline uint32_t filter2count_range_mod3() {
		static_assert(k_upper <= length);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		constexpr T mask = (T(1u) << (2u*k_upper)%bits_per_limb) - 1u;
		constexpr uint32_t limb = MAX(1, (k_upper+numbers_per_limb-1)/ numbers_per_limb);

		if constexpr (limb == 1) {
			return __builtin_popcountll(__data[0]&mask&m);
		}

		uint32_t ctr = 0;
		#pragma unroll
		for (uint32_t i = 0; i < limb-1; ++i) {
			ctr += __builtin_popcountll(__data[i]&m);
		}

		return ctr + __builtin_popcountll(__data[limb-1]&mask&m);
	}

	template<const uint16_t k_upper>
	static inline uint32_t filter2count_range_mod3_limb(const T a) noexcept {
		ASSERT(0 < k_upper && k_upper <= length);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		constexpr T mm = (T(1u) << (2u*k_upper)) - 1;
		constexpr T mask =  m & mm;

		return __builtin_popcountll(a&mask);
	}

	/// checks whether an element needs be filtered or not.
	/// \param limit number of two allowed in the container.
	/// \return true if the element contains more than `limit` twos
	///			false otherwise
	inline bool filter2_mod3(const uint32_t limit) const noexcept {
		uint32_t ctr = 0;
		for (uint32_t i = 0; i < internal_limbs; ++i) {
			ctr += filter2count_mod3_limb(__data[i]);
			if (ctr > limit)
				return true;
		}

		return false;
	}

	inline static void times2_mod3(kAryPackedContainer_T &v3, kAryPackedContainer_T &v1) noexcept {
		uint32_t i = 0;
		for (; i < internal_limbs; i++){
			v3.__data[i] = times2_mod3_limb(v1.__data[i]);
		}
	}

	/// TODO explain
	/// \param v3
	/// \param v1
	/// \param v2
	inline static void add(kAryPackedContainer_T &v3,
	                       kAryPackedContainer_T const &v1,
	                       kAryPackedContainer_T const &v2) noexcept {
		if constexpr (internal_limbs == 1) {
			v3.__data[0] = add_mod3_limb(v1.__data[0], v2.__data[0]);
			return;
		} else if constexpr (internal_limbs == 2) {
			__uint128_t t = add_mod3_limb128(*((__uint128_t *)v1.__data.data()), *((__uint128_t *)v2.__data.data()));
			*(__uint128_t *)v3.__data.data() = t;
			return;
		}

		uint32_t i = 0;
#ifdef USE_AVX2
		for (; i+4 <= internal_limbs; i += 4) {
			__m256i t = add_mod3_limb256(_mm256_lddqu_si256((__m256i *)&v1.__data[i]), _mm256_lddqu_si256((__m256i *)&v2.__data[i]));
			_mm256_storeu_si256((__m256i *)&v3.__data[i], t);
		}

#endif
		for (; i+2 <= internal_limbs; i += 2) {
			__uint128_t t = add_mod3_limb128(*((__uint128_t *)&v1.__data[i]), *((__uint128_t *)&v2.__data[i]));
			*((__uint128_t *)&v3.__data[i]) = t;
		}

		for (; i < internal_limbs; i++) {
			v3.__data[i] = add_mod3_limb(v1.__data[i], v2.__data[i]);
		}
	}

	// generic add: v3 = v1 + v2 between k_lower and k_upper
	inline static void add(kAryPackedContainer_T &v3, kAryPackedContainer_T const &v1, kAryPackedContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			DataType data = v1.get(i) + v2.get(i);
			v3.set(data % MOD, i);
		}
	}

	// generic add: v3 = v1 + v2 between k_lower and k_upper with filter method
	inline static bool add(kAryPackedContainer_T &v3, kAryPackedContainer_T const &v1, kAryPackedContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper, const uint32_t filter2) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			DataType data = v1.get(i) + v2.get(i);
			v3.set(data % MOD, i);
		}

		return v3.filter2_mod3(filter2);
	}

	inline static void sub(kAryPackedContainer_T &v3, kAryPackedContainer_T const &v1, kAryPackedContainer_T const &v2) noexcept {
		if constexpr (internal_limbs == 1) {
			v3.__data[0] = sub_mod3_limb(v1.__data[0], v2.__data[0]);
			return;
		}
		else if constexpr (internal_limbs == 2) {
			__uint128_t t = sub_mod3_limb128(*((__uint128_t *)v1.__data.data()), *((__uint128_t *)v2.__data.data()));
			*((__uint128_t *)v3.__data.data()) = t;
			return;
		}

		uint32_t i = 0;
#ifdef USE_AVX2
		for (; i+4 <= internal_limbs; i += 4) {
			__m256i t = sub_mod3_limb256(_mm256_lddqu_si256((__m256i *)&v1.__data[i]), _mm256_lddqu_si256((__m256i *)&v2.__data[i]));
			_mm256_storeu_si256((__m256i *)&v3.__data[i], t);
		}
#endif
		for (; i+2 <= internal_limbs; i += 2) {
			__uint128_t t = sub_mod3_limb128(*((__uint128_t *)&v1.__data[i]), *((__uint128_t *)&v2.__data[i]));
			*((__uint128_t *)&v3.__data[i]) = t;
		}

		for (; i < internal_limbs; i++) {
			v3.__data[i] = sub_mod3_limb(v1.__data[i], v2.__data[i]);
		}
	}

	inline static void sub(kAryPackedContainer_T &v3, kAryPackedContainer_T const &v1, kAryPackedContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			int64_t data = int64_t(v1.get(i)) - int64_t(v2.get(i));
			if (data < 0)
				data += MOD;
			v3.set(data % MOD, i);
		}
	}

	inline static bool cmp(kAryPackedContainer_T const &v1, kAryPackedContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			if (v1.get(i) != v2.get(i))
				return false;
		}
		return true;
	}

	inline static void set(kAryPackedContainer_T &v1, kAryPackedContainer_T const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			v1.set(v2.get(i), i);
		}
	}

	bool is_equal(kAryPackedContainer_T const &obj) const noexcept {
		return cmp(*this, obj, 0, length);
	}

	bool is_equal(kAryPackedContainer_T const &obj, const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	bool is_greater(kAryPackedContainer_T const &obj, const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_upper; i > k_lower; i++) {
			if (get(i-1) < obj.get(i-1))
				return false;
		}

		return true;
	}

	bool is_lower(kAryPackedContainer_T const &obj, const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		for (unsigned int i = k_lower; i < k_upper; i++) {
			if(get(i) > obj.get(i))
				return false;
		}
		return true;
	}

	// add on full length and return the weight only between [l, h]
	static uint16_t add_only_upper_weight_partly(kAryPackedContainer_T &v3,
	                                             kAryPackedContainer_T &v1,
	                                             kAryPackedContainer_T &v2,
	                                             const uint32_t l, const uint32_t h) noexcept {
		uint16_t weight = 0;
		add(v3, v1, v2);
		for (uint32_t i = l; i < h; ++i) {
			weight += v3.get(i) != 0;
		}

		return weight;
	}

	// optimised version of the function above.
	template<const uint32_t l, const uint32_t h>
	static uint16_t add_only_weight_partly(kAryPackedContainer_T &v3,
	                                       kAryPackedContainer_T &v1,
	                                       kAryPackedContainer_T &v2) noexcept {
		constexpr uint32_t llimb = l/numbers_per_limb;
		constexpr uint32_t hlimb = h/numbers_per_limb;
		constexpr T lmask = ~((T(1u) << (l*bits_per_number)) - 1);
		constexpr T hmask =  ( T(1u) << (h*bits_per_number)) - 1;
		uint16_t weight = 0;

		// first add the lower limbs
		for (uint32_t i = 0; i < llimb; i++) {
			v3.__data[i] = add_mod3_limb(v1.__data[i], v2.__data[i]);
		}

		// add the llimb with weight
		v3.__data[llimb] = add_mod3_limb(v1.__data[llimb], v2.__data[llimb]);
		weight = hammingweight_mod3_limb(v3.__data[llimb]&lmask);

		// add the limbs betwen l and h
		for (uint32_t i = llimb+1; i < hlimb; i++) {
			v3.__data[i] = add_mod3_limb(v1.__data[i], v2.__data[i]);
			weight = hammingweight_mod3_limb(v3.__data[i]);
		}

		// add the high limb
		v3.__data[hlimb] = add_mod3_limb(v1.__data[hlimb], v2.__data[hlimb]);
		weight = hammingweight_mod3_limb(v3.__data[hlimb]&hmask);

		// add everything that's is left
		for (uint32_t i = hlimb+1; i < internal_limbs; i++) {
			v3.__data[i] = add_mod3_limb(v1.__data[i], v2.__data[i]);
		}

		return weight;
	}

	void left_shift(const uint32_t i) noexcept {
		for (uint32_t j = length; j > i; j--) {
			set(get(j-i-1), j-1);
		}

		// clear the rest.
		for (uint32_t j = 0; j < i; j++) {
			set(0, j);
		}
	}


	DataType operator [](size_t i) noexcept {
		ASSERT(i < length && "wrong access index");
		return get(i);
	}

	DataType operator [](const size_t i) const noexcept {
		ASSERT(i < length && "wrong access index");
		return get(i);
	};

	kAryPackedContainer_T& operator =(kAryPackedContainer_T const &obj) noexcept {
		ASSERT(size() == obj.size() && "äh?");

		if (likely(this != &obj)) { // self-assignment check expected
			__data = obj.__data;
		}

		return *this;
	}

	void print() const {
		print(0, length);
	}

	// prints ternary
	void print(const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_lower < length && k_upper <= length && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << unsigned(get(i));
		}
		std::cout << "\n";
	}

	// prints also ternary
	template<typename TT>
	static void print(const TT in) noexcept {
		constexpr static TT mask = (TT(1u) << bits_per_number)-1;
		TT data = in;
		for (int i = 0; i < (sizeof(TT)*8/bits_per_number); ++i) {
			std::cout << unsigned(data&mask);
			data >>= bits_per_number;
		}

		std::cout << "\n";
	}

#ifdef USE_AVX2
	static void print(const __m256i t) noexcept {
		constexpr static uint64_t mask = (uint64_t (1u) << bits_per_number)-1;
		int64_t datas[4] = {_mm256_extract_epi64(t, 0), _mm256_extract_epi64(t, 1),
							_mm256_extract_epi64(t, 2), _mm256_extract_epi64(t, 3)};

		for (uint32_t j = 0; j < 4; j++) {
			uint64_t data = datas[j];
			for (int i = 0; i < (sizeof(uint64_t)*8/bits_per_number); ++i) {
				std::cout << unsigned(data & mask);
				data >>= bits_per_number;
			}
		}
		std::cout << "\n";
	}
#endif


	// print the `pos
	void static print_binary(const uint64_t data,
	                         const uint16_t k_lower,
	                         const uint16_t k_higher) noexcept {
		uint64_t d = data;
		for (int i = k_lower; i < k_higher; ++i) {
			std::cout << unsigned(d&1);
			std::cout << unsigned((d>>1)&1);
			d >>= 2;
		}
		std::cout << "\n";
	}

	// print the `pos
	void print_binary(const uint16_t k_lower, const uint16_t k_higher) const noexcept {
		for (int i = k_lower; i < k_higher; ++i) {
			std::cout << unsigned(get(i)&1);
			std::cout << unsigned((get(i)>>1)&1);
		}
		std::cout << "\n";
	}

	// iterators.
	auto begin() noexcept { return __data.begin();}
	const auto begin() const noexcept { return __data.begin();}
	auto end() noexcept { return __data.end();}
	const auto end() const noexcept { return __data.end();}


	// return `true` if the datastruct contains binary data.
	__FORCEINLINE__ constexpr static bool binary() noexcept { return false; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return length; }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return internal_limbs; }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return internal_limbs*sizeof(T); }

	std::array<T, internal_limbs>& data() noexcept { return __data; }
	const std::array<T, internal_limbs>& data() const noexcept { return __data; }
	//T& data(const size_t index) { ASSERT(index < length && "wrong index"); return __data[index]; }
	const T data(const size_t index) const noexcept { ASSERT(index < length && "wrong index"); return __data[index]; }
	T limb(const size_t index) { ASSERT(index < length && "wrong index"); return __data[index]; }
	const T limb(const size_t index) const noexcept { ASSERT(index < length && "wrong index"); return __data[index]; }

	// get raw access to the underlying data.
	T* ptr() noexcept { return __data.data(); }
	const T* ptr() const noexcept { return __data.data(); }

	// TODO remove
	T get_type() noexcept {return __data[0]; }
};

///

/// \tparam length
/// \tparam LimbType
template<uint32_t length, typename LimbType=uint64_t>
	requires BinaryContainerAble<LimbType>
class BinaryContainer {
public:

	// Internal Types needed for the template system.
	typedef LimbType ContainerLimbType;
	typedef bool DataType;

	// internal data length. Need to export it for the template system.
	constexpr static uint32_t LENGTH = length;
	constexpr static LimbType minus_one = LimbType(-1);
private:
	constexpr static uint64_t popcount(LimbType x) noexcept {
		if constexpr(limb_bits_width() == 64) {
			return __builtin_popcountll(x);
		} else {
			return __builtin_popcountl(x);
		}
	}

	constexpr static uint64_t popcount(LimbType x, LimbType mask) noexcept {
		if constexpr(limb_bits_width() == 64) {
			return __builtin_popcountll(x & mask);
		} else {
			return __builtin_popcountl(x & mask);
		}
	}

public:
	// how many limbs to we need and how wide are they.
	constexpr static uint16_t limb_bits_width() noexcept { return limb_bytes_width() * 8; };
	constexpr static uint16_t limb_bytes_width() noexcept { return sizeof(LimbType); };

private:
	// DO NOT CALL THIS FUNCTION. Use 'limbs()'.
	constexpr static uint16_t compute_limbs() noexcept {
#ifdef BINARY_CONTAINER_ALIGNMENT
		return (alignment()+limb_bits_width()-1)/limb_bits_width();
#else
		return (length+limb_bits_width()-1)/limb_bits_width();
#endif
	};

public:

	/// default constructor
	BinaryContainer() noexcept : __data() { ASSERT(length > 0); }

	/// Copy Constructor
	BinaryContainer(const BinaryContainer& a)  noexcept : __data(a.__data) {}


	// round a given amount of 'in' bits to the nearest limb excluding the the lowest overflowing bits
	// eg 13 -> 64
	constexpr static uint16_t round_up(uint16_t in) noexcept { return round_up_to_limb(in) * limb_bits_width(); }
 	constexpr static uint16_t round_up_to_limb(uint16_t in) noexcept {return (in/limb_bits_width())+1; }

	// the same as above only rounding down
	// 13 -> 0
	constexpr static uint16_t round_down(uint16_t in) noexcept { return round_down_to_limb(in) * limb_bits_width(); }
	constexpr static const uint16_t round_down_to_limb(uint16_t in) { return (in/limb_bits_width()); }

	// calculate from a bit position 'i' the mask to set it.
	constexpr static LimbType mask(uint16_t i) noexcept {
		ASSERT(i <= length && "wrong access index");
		LimbType u = i%limb_bits_width();
		return (LimbType(1) << u);
	}

	// same as the function below, but catches the special case when i == 0 %64.
	constexpr static LimbType lower_mask2(const uint16_t i) noexcept {
		ASSERT(i <= length);
		LimbType u = i%limb_bits_width();
		if (u == 0) return LimbType(-1);
		return ((LimbType(1) << u) - 1);
	}

	// given the i-th bit this function will return a bits mask where the lower 'i' bits are set. Everything will be
	// realigned to limb_bits_width().
	constexpr static LimbType lower_mask(const uint16_t i) noexcept {
		ASSERT(i <= length);
		return ((LimbType(1) << (i%limb_bits_width())) - 1);
	}

	// given the i-th bit this function will return a bits mask where the higher (n-i)bits are set.
	constexpr static LimbType higher_mask(const uint16_t i) noexcept {
		ASSERT(i <= length);
		return (~((LimbType(1) << (i%limb_bits_width())) - 1));
	}

	// given the i-th bit this function will return a bits mask where the lower 'n-i' bits are set. Everything will be
	// realigned to limb_bits_width().
	constexpr static LimbType lower_mask_inverse(const uint16_t i) noexcept
	{
		ASSERT(i <= length && "wrong access index");
		LimbType u = i%limb_bits_width();

		if (u == 0)
			return -1;

		auto b = (LimbType(1) << (limb_bits_width()-u));
		return b - LimbType(1);
	}

	// given the i-th bit this function will return a bits mask where the higher (i) bits are set.
	constexpr static LimbType higher_mask_inverse(const uint16_t i) noexcept {
		ASSERT(i <= length && "wrong access index");
		return (~lower_mask_inverse(i));
	}

	// not shifted.
	constexpr LimbType get_bit(const uint16_t i) const noexcept {
		return __data[round_down_to_limb(i)] & mask(i);
	}

	// shifted.
	constexpr bool get_bit_shifted(const uint16_t i) const noexcept {
		return (__data[round_down_to_limb(i)] & mask(i)) >> i;
	}

	// return the bits [i,..., j) in one limb
	inline LimbType get_bits(const uint16_t i, const uint16_t j) const noexcept {
		ASSERT(j > i && j-i <= limb_bits_width() && j <= length);
        const LimbType lmask = higher_mask(i);
		const LimbType rmask = lower_mask2(j);
		const int64_t lower_limb = i/limb_bits_width();
		const int64_t higher_limb = (j-1)/limb_bits_width();

		const uint64_t shift = i%limb_bits_width();
		if (lower_limb == higher_limb) {
			return (__data[lower_limb] & lmask & rmask) >> (shift);
		} else {
			const LimbType a = __data[lower_limb] & lmask;
			const LimbType b = __data[higher_limb] & rmask;

			auto c = (a >> shift);
			auto d = (b << (limb_bits_width()-shift));
			auto r = c ^ d;
			return  r ;
		}
	}

	/// call like this
	/// const LimbType lmask = higher_mask(i);
	/// const LimbType rmask = lower_mask2(j);
	/// const int64_t lower_limb = i / limb_bits_width();
	/// const int64_t higher_limb = (j - 1) / limb_bits_width();
	/// const uint64_t shift = i % limb_bits_width();
	inline LimbType get_bits(const uint64_t llimb, const uint64_t rlimb, const uint64_t lmask, const uint64_t rmask, const uint64_t shift) const noexcept {
		if (llimb == rlimb) {
			return (__data[llimb] & lmask & rmask) >> (shift);
		} else {
			const LimbType a = __data[llimb] & lmask;
			const LimbType b = __data[rlimb] & rmask;

			auto c = (a >> shift);
			auto d = (b << (limb_bits_width()-shift));
			auto r = c ^ d;
			return r;
		}
	}

	inline constexpr void write_bit(const uint16_t pos, const uint8_t bit) noexcept {
#define __WRITE_BIT(w, spot, value) ((w) = (((w) & ~(m4ri_one << (spot))) | (__M4RI_CONVERT_TO_WORD(value) << (spot))))
		__WRITE_BIT(__data[pos/m4ri_radix], pos%m4ri_radix, bit);
	}

	inline constexpr void set_bit(const uint16_t pos) noexcept {
		__data[round_down_to_limb(pos)] |= (LimbType(1) << (pos));
	}

	inline constexpr void flip_bit(const uint16_t pos) noexcept {
		__data[round_down_to_limb(pos)] ^= (uint64_t(1) << (pos));
	}

	inline constexpr void clear_bit(const uint16_t pos) noexcept {
		__data[round_down_to_limb(pos)] &= ~(uint64_t(1) << (pos));
	}

	/// zero the complete data vector
	constexpr void zero() noexcept {
		LOOP_UNROLL();
		for (unsigned int i = 0; i < limbs(); ++i) {
			__data[i] = 0;
		}
	}

	constexpr void zero(const uint32_t k_lower, const uint32_t k_upper) noexcept {
		const uint64_t lower = round_down_to_limb(k_lower);
		const uint64_t upper = round_down_to_limb(k_upper);

		const LimbType lmask = higher_mask(k_lower);
		const LimbType umask = k_upper%64 == 0 ? LimbType(0) : lower_mask(k_upper);

		if (lower == upper) {
			const LimbType mask = ~(lmask&umask);
			__data[lower] &= mask;
			return ;
		}

		__data[lower] &= ~lmask;
		__data[upper] &= ~umask;

		LOOP_UNROLL();
		for (unsigned int i = lower+1; i < upper; ++i) {
			__data[i] = 0;
		}
	}

	// seth the whole array to 'fff...fff'
	void one() noexcept {
		for (int i = 0; i < limbs(); ++i) {
			__data[i] = ~(__data[i] & 0);
		}
	}

	void one(const uint32_t k_lower, const uint32_t k_upper) noexcept {
		const uint64_t lower = round_down_to_limb(k_lower);
		const uint64_t upper = round_down_to_limb(k_upper);

		const LimbType lmask = higher_mask(k_lower);
		const LimbType umask = k_upper%64 == 0 ? LimbType(0) : lower_mask(k_upper);

		if (lower == upper) {
			const LimbType mask = (lmask&umask);
			__data[lower] |= minus_one & mask;
			return ;
		}

		__data[lower] |= minus_one & lmask;
		__data[upper] |= minus_one & umask;

		for (int i = lower+1; i < upper; ++i) {
			__data[i] = ~(__data[i] & 0);
		}
	}

	LimbType static random_limb() noexcept {
		return fastrandombytes_uint64();
	}

	LimbType static random_limb(const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper > k_lower && k_upper - k_lower < 64);

		const LimbType lmask = higher_mask(k_lower);
		const LimbType umask = k_upper%64 == 0 ? LimbType(-1) : lower_mask(k_upper%64);

		return fastrandombytes_uint64() & lmask & umask;
	}

	/// split the full length BinaryContainer into `k` windows. Inject in every window weight `w` on random positions.
	void random_with_weight_per_windows(const uint64_t w, const uint64_t k) noexcept {
		std::vector<uint64_t> buckets_windows{};

		// this stupid approach needs to be done, because if w is not dividing n the last bits would be unused.
		buckets_windows.resize(k+1);
		for (int i = 0; i < k; ++i) {
			buckets_windows[i] = i*length/k;
		}
		buckets_windows[k] = length;

		// clear everything.
		zero();

		// for every window.
		for (int i = 0; i < k; ++i) {
			uint64_t cur_offset = buckets_windows[i];
			uint64_t windows_length = buckets_windows[i+1] - buckets_windows[i];

			for (int j = 0; j < w; ++j) {
				write_bit(cur_offset + j, true);
			}

			// now permute
			for (int l = 0; l < windows_length; ++l) {
				uint64_t pos = random_limb() % (windows_length - l);
				auto t = get_bit_shifted(cur_offset + l);
				write_bit(cur_offset + l, get_bit_shifted(cur_offset+l+pos));
				write_bit(cur_offset+l+pos, t);
			}
		}
	}

	void random_with_weight(const uint64_t w) noexcept {
		zero();

		for (int i = 0; i < w; ++i) {
			write_bit(i, true);
		}

		// now permute
		for (int i = 0; i < length; ++i) {
			uint64_t pos = random_limb() % (length - i);
			bool t = get_bit_shifted(i);
			write_bit(i, get_bit_shifted(i+pos));
			write_bit(i+pos, t);
		}
	}

	/// set the whole data array on random data.
	void random() noexcept {
		constexpr uint64_t apply_mask = length%limb_bits_width()==0 ? lower_mask(length)-1 : lower_mask(length);

		if constexpr (length < 64) {
			__data[0] = fastrandombytes_uint64() & apply_mask;
		} else {
			for (uint32_t i = 0; i < limbs()-1; ++i) {
				__data[i] = fastrandombytes_uint64();
			}
			__data[limbs()-1] = fastrandombytes_uint64() & apply_mask;
		}
	}

	bool is_zero() const noexcept {
		for (uint32_t i = 0; i < limbs(); ++i) {
			if(__data[i] != 0)
				return false;
		}

		return true;
	}

	// returns the position in which bits are set.
	void get_bits_set(uint32_t *P, const uint16_t pos=1) const noexcept {
		uint16_t ctr = 0;
		for (uint32_t i = 0; i < length; ++i) {
			if (get_bit(i)) {
				P[ctr++] = i;
				if(ctr == pos)
					return;
			}
		}
	}

	// M4RI (method Of The 4 Russians) glue code.
	// export/import function
	void to_m4ri(word *a) const noexcept {
		a = __data.data();
	}
	void column_from_m4ri(const mzd_t *H, const uint32_t col) noexcept {
		ASSERT(uint64_t(H->ncols) > uint64_t(col));
		for (int i = 0; i < H->nrows; ++i) {
			write_bit(i, mzd_read_bit(H, i, col));
		}
	}
	void column_from_m4ri(const mzd_t *H, const uint32_t col, const uint32_t srow) noexcept {
		ASSERT(uint64_t(H->ncols) > uint64_t(col));
		for (int i = srow; i < H->nrows; ++i) {
			write_bit(i-srow, mzd_read_bit(H, i, col));
		}
	}
	void from_m4ri(const word *a) noexcept {
		for (uint32_t i = 0; i < limbs(); ++i) {
			__data[i] = a[i];
		}
	}
	void to_m4ri(mzd_t *a) noexcept {
		ASSERT(a->nrows == 1 && a->ncols > 0);
		to_m4ri(a->rows[0]);
	}
	void from_m4ri(const mzd_t *a) noexcept {
		ASSERT(a->nrows == 1 && a->ncols > 0);
		from_m4ri(a->rows[0]);
	}

	// swap the two bits i, j
	void swap(const uint16_t i, const uint16_t j) noexcept {
		ASSERT(i < length && j < length);
		auto t = get_bit_shifted(i);
		write_bit(i, get_bit_shifted(j));
		write_bit(j, t);
	}

	void flip(const uint16_t i) noexcept {
		ASSERT(i < length);
		__data[round_down_to_limb(i)] ^= mask(i);
	}

	inline void neg(const uint16_t k_lower, const uint16_t k_upper) noexcept {
		// do nothing.
	}

	// full length addition.
	inline int add(BinaryContainer const &v) noexcept {
		add(*this, *this, v);
		return 0;
	}

	// windowed addition.
	inline void add(BinaryContainer const &v, const uint32_t k_lower, const uint32_t k_upper) noexcept {
		add(*this, *this, v, k_lower, k_upper);
	}

	constexpr static void add(LimbType *v5, LimbType const *v1, LimbType const *v2, LimbType const *v3, LimbType const *v4, const uint32_t limbs) noexcept {
		int64_t i = 0;

#ifdef USE_AVX2
		LOOP_UNROLL()
		for (; i+4 <= limbs; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD((float*)v1 + 2*i);
			__m256 y_avx = MM256_LOAD((float*)v2 + 2*i);
			__m256 y1_avx = MM256_LOAD((float*)v3 + 2*i);
			__m256 y2_avx = MM256_LOAD((float*)v4 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			_mm256_xor_ps(z_avx, y1_avx);
			_mm256_xor_ps(z_avx, y2_avx);

			MM256_STORE((float*)v5 + 2*i, z_avx);
		}
#endif

		for (; i < limbs; ++i) {
			v5[i] = v1[i] ^ v2[i] ^ v3[i] ^ v4[i];
		}
	}

	constexpr static void add(LimbType *v4, LimbType const *v1, LimbType const *v2, LimbType const *v3, const uint32_t limbs) noexcept {
		int64_t i = 0;

#ifdef USE_AVX2
		LOOP_UNROLL()
		for (; i+4 <= limbs; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD((float*)v1 + 2*i);
			__m256 y_avx = MM256_LOAD((float*)v2 + 2*i);
			__m256 y1_avx = MM256_LOAD((float*)v3 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			_mm256_xor_ps(z_avx, y1_avx);
			MM256_STORE((float*)v4 + 2*i, z_avx);
		}
#endif

		for (; i < limbs; ++i) {
			v4[i] = v1[i] ^ v2[i] ^ v3[i];
		}
	}

	// full length addition
	constexpr static void add(LimbType *v3, LimbType const *v1, LimbType const *v2, const uint32_t limbs) noexcept {
		int64_t i = 0;

#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < limbs/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1 + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2 + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3 + i, z_avx);
		}
		i*=4;
#else

		LOOP_UNROLL()
		for (; i+4 <= limbs; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD((float*)v1 + 2*i);
			__m256 y_avx = MM256_LOAD((float*)v2 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			MM256_STORE((float*)v3 + 2*i, z_avx);
		}
#endif
#endif

		for (; i < limbs; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}
	}

	// full length addition
	constexpr static void add(LimbType *v3, LimbType const *v1, LimbType const *v2) noexcept {
		int64_t i = 0;

#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < limbs()/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1 + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2 + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3 + i, z_avx);
		}
		i*=4;
#else
		LOOP_UNROLL()
		for (; i+4 < limbs(); i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD_UNALIGNED((float*)v1 + 2*i);
			__m256 y_avx = MM256_LOAD_UNALIGNED((float*)v2 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			MM256_STORE_UNALIGNED((float*)v3 + 2*i, z_avx);
		}
#endif
#endif

		for (; i < limbs(); ++i) {
			v3[i] = v1[i] ^ v2[i];
		}
	}

	static void add_withoutasm(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
	}

	//  IMPORTANT: this function does a full length addition
	__FORCEINLINE__ static void add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		constexpr uint32_t upper = limbs();
		uint32_t i = 0;
#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < upper/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1.ptr() + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2.ptr() + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3.ptr() + i, z_avx);
		}
		i*=4;
#else
		LOOP_UNROLL()
		for (; i+4 < upper; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD_UNALIGNED((float*)v1.ptr() + 2*i);
			__m256 y_avx = MM256_LOAD_UNALIGNED((float*)v2.ptr() + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			MM256_STORE_UNALIGNED((float*)v3.ptr() + 2*i, z_avx);
		}
#endif
#endif
		LOOP_UNROLL()
		for (; i < upper; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
	}

	// add between the coordinate l, h
	template<const uint32_t k_lower, const uint32_t k_upper>
	__FORCEINLINE__ static void add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		constexpr LimbType lmask        = higher_mask(k_lower%limb_bits_width());
		constexpr LimbType rmask        = lower_mask2(k_upper%limb_bits_width());
		constexpr uint32_t lower_limb   = k_lower / limb_bits_width();
		constexpr uint32_t higher_limb  = (k_upper-1) / limb_bits_width();

		static_assert(k_lower < k_upper);
		static_assert(k_upper <= length);
		static_assert(higher_limb <= limbs());

		if constexpr(lower_limb == higher_limb) {
			constexpr LimbType mask = k_upper%64u == 0u ? lmask : (lmask & rmask);
			LimbType tmp1 = (v3.__data[lower_limb] & ~(mask));
			LimbType tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			return;
		}

		LOOP_UNROLL();
		for (uint32_t i = lower_limb+1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		LimbType tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		LimbType tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		LimbType tmp11 = (v3.__data[lower_limb] & ~(lmask));
		LimbType tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1^tmp11;
		v3.__data[higher_limb]= tmp2^tmp21;

//		uint32_t i = 0;
//#ifdef USE_AVX2
//#ifdef USE_AVX2_SPECIAL_ALIGNMENT
//		LOOP_UNROLL()
//		for (; i < upper/4; i++) {
//			__m256i x_avx = _mm256_load_si256((__m256i *)v1.ptr() + i);
//			__m256i y_avx = _mm256_load_si256((__m256i *)v2.ptr() + i);
//			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
//			_mm256_store_si256((__m256i *)v3.ptr() + i, z_avx);
//		}
//		i*=4;
//#else
//		LOOP_UNROLL()
//		for (; i+4 < upper; i+=4) {
//			// we need to access the memory unaligned
//			__m256 x_avx = MM256_LOAD_UNALIGNED((float*)v1.ptr() + 2*i);
//			__m256 y_avx = MM256_LOAD_UNALIGNED((float*)v2.ptr() + 2*i);
//			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
//			MM256_STORE_UNALIGNED((float*)v3.ptr() + 2*i, z_avx);
//		}
//#endif
//#endif
//		LOOP_UNROLL()
//		for (; i < upper; ++i) {
//			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
//		}
	}

	/// same as the function below.
	template<const uint32_t llimb, const uint32_t ulimb, const LimbType lmask, const LimbType rmask>
	static void add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		if constexpr (llimb == ulimb) {
			constexpr LimbType mask = (lmask & rmask);
			LimbType tmp1 = (v3.__data[llimb] & ~(mask));
			LimbType tmp2 = (v1.__data[llimb] ^ v2.__data[llimb]) & mask;
			v3.__data[llimb] = tmp1 ^ tmp2;
			return;
		}

		uint32_t i = llimb+1;

#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < ulimb/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1.__data.data() + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2.__data.data() + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3.__data.data() + i, z_avx);
		}
		i*=4;
#else
		LOOP_UNROLL()

		for (; i+4 < ulimb; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD_UNALIGNED((float*)v1.__data.data() + 2*i);
			__m256 y_avx = MM256_LOAD_UNALIGNED((float*)v2.__data.data() + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			MM256_STORE_UNALIGNED((float*)v3.__data.data() + 2*i, z_avx);
		}
#endif
#endif

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		LimbType tmp1 = (v1.__data[llimb] ^ v2.__data[llimb]) & lmask;
		LimbType tmp2 = (v1.__data[ulimb] ^ v2.__data[ulimb]) & rmask;
		LimbType tmp11 = (v3.__data[llimb] & ~(lmask));
		LimbType tmp21 = (v3.__data[ulimb] & ~(rmask));

		v3.__data[llimb] = tmp1^tmp11;
		v3.__data[ulimb] = tmp2^tmp21;
	}



	/// Another heavy overloaded function to add two vectors. Add two vector v1+v3=v3 and safe the result in v3.
	/// But only calculates the sum between the limbs `llimb` and `ulimb` while apply to these limbs the masks
	/// \tparam llimb 	lowest limb
	/// \tparam ulimb 	highest limb
	/// \tparam lmask	bit mask for llimb
	/// \tparam rmask	bit mask for ulimb
	/// \return nothing
	template<const uint32_t llimb, const uint32_t ulimb, const LimbType lmask, const LimbType rmask>
	constexpr static void add(LimbType *v3, LimbType const *v1, LimbType const *v2) noexcept {
		if constexpr (llimb == ulimb) {
			constexpr LimbType mask = (lmask & rmask);
			LimbType tmp1 = (v3[llimb] & ~(mask));
			LimbType tmp2 = (v1[llimb] ^ v2[llimb]) & mask;
			v3[llimb] = tmp1 ^ tmp2;
			return;
		}

		int32_t i = llimb+1;

#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < ulimb/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1 + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2 + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3 + i, z_avx);
		}
		i*=4;
#else
		LOOP_UNROLL()
		for (; i+4 < ulimb; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD((float*)v1 + 2*i);
			__m256 y_avx = MM256_LOAD((float*)v2 + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			MM256_STORE((float*)v3 + 2*i, z_avx);
		}
#endif
#endif

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}

		LimbType tmp1 = (v1[llimb] ^ v2[llimb]) & lmask;
		LimbType tmp2 = (v1[ulimb] ^ v2[ulimb]) & rmask;
		LimbType tmp11 = (v3[llimb] & ~(lmask));
		LimbType tmp21 = (v3[ulimb] & ~(rmask));

		v3[llimb] = tmp1^tmp11;
		v3[ulimb] = tmp2^tmp21;
	}

	/// Does a full length addition. But the hamming weight is just calculated up to `ulimb` with the mask `rmask`
	/// \tparam ulimb	max limb to calc the hamming weight
	/// \tparam rmask	mask to apply before calc the weight.
	/// \return
	template<const uint32_t ulimb,const LimbType rmask>
	inline static uint32_t add_only_upper_weight_partly(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		int32_t i = 0;
		uint32_t hm = 0;

// AVX optimisation disable, because i dont know a good way to calculate the hamming weight of one
#ifdef USE_AVX2
		__m256i acc = _mm256_setzero_si256();
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < ulimb/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1.ptr() + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2.ptr() + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			acc = _mm256_add_epi64(acc, _mm256_sad_epu8(hammingweight_mod2_limb256_nonacc(z_avx), _mm256_setzero_si256()));
			_mm256_store_si256((__m256i *)v3.ptr() + i, z_avx);
		}

		i*=4;
#else
        LOOP_UNROLL()
        for (; i+4 < ulimb; i+=4) {
            // we need to access the memory unaligned
            __m256 x_avx = MM256_LOAD((float*)v1.ptr() + 2*i);
            __m256 y_avx = MM256_LOAD((float*)v2.ptr() + 2*i);
            __m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
	        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(hammingweight_mod2_limb256_nonacc((__m256i)z_avx), _mm256_setzero_si256()));
	        MM256_STORE((float*)v3.ptr() + 2*i, z_avx);
        }
#endif
		hm += (uint64_t)(_mm256_extract_epi64(acc, 0));
		hm += (uint64_t)(_mm256_extract_epi64(acc, 1));
		hm += (uint64_t)(_mm256_extract_epi64(acc, 2));
		hm += (uint64_t)(_mm256_extract_epi64(acc, 3));
#endif

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcount(v3.__data[i]);
		}

		v3.__data[ulimb] = v1.__data[ulimb] ^ v2.__data[ulimb];
		return hm + popcount(v3.__data[ulimb] & rmask);
	}

	template<const uint32_t ulimb,const LimbType rmask>
	inline static uint32_t add_only_upper_weight_partly_withoutasm(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		uint32_t i = 0;
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcount(v3.__data[i]);
		}

		for (; i < limbs(); i++) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
		return hm + popcount(v3.__data[ulimb] & rmask);
	}

	template<const uint32_t j>
	inline static uint32_t add_only_upper_weight_partly_withoutasm(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		constexpr uint32_t ulimb         = round_down_to_limb(j - 1);
		constexpr static LimbType rmask = lower_mask2(j);

		int32_t i = 0;
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcount(v3.__data[i]);
		}

		for (; i < limbs(); i++) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
		return hm + popcount(v3.__data[ulimb] & rmask);
	}

	template<const uint32_t ulimb,const LimbType rmask, const uint32_t early_exit>
	static uint32_t add_only_upper_weight_partly_withoutasm_earlyexit(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (uint32_t i = 0; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcount(v3.__data[i]);
			if (hm > early_exit)
				return hm;
		}

		v3.__data[ulimb] = v1.__data[ulimb] ^ v2.__data[ulimb];
		return hm + popcount(v3.__data[ulimb] & rmask);
	}

	// TODO optimize
	template<const uint32_t ulimb,const LimbType rmask, const uint32_t early_exit>
	static uint32_t add_only_upper_weight_partly_earlyexit(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		uint32_t hm = 0;
		uint32_t i = 0;
#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < ulimb/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1.__data.data() + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2.__data.data() + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3.__data.data() + i, z_avx);
		}
		i*=4;
#else
		LOOP_UNROLL()

		for (; i+4 < ulimb; i+=4) {
			// we need to access the memory unaligned
			__m256 x_avx = MM256_LOAD_UNALIGNED((float*)v1.__data.data() + 2*i);
			__m256 y_avx = MM256_LOAD_UNALIGNED((float*)v2.__data.data() + 2*i);
			__m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
			MM256_STORE_UNALIGNED((float*)v3.__data.data() + 2*i, z_avx);
		}
#endif
#endif

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		LOOP_UNROLL();
		for (uint32_t i = 0; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcount(v3.__data[i]);
			if (hm > early_exit)
				return hm;
		}

		v3.__data[ulimb] = v1.__data[ulimb] ^ v2.__data[ulimb];
		return hm + popcount(v3.__data[ulimb] & rmask);
	}

	/// calculates the sumf of v3= v1+v2 on the full length, but return the weight of v3 only ont the first coordinate
	/// defined by `ulimb` and `rmask`
	/// \tparam ulimb	max limb to calculate the weight on the full length of each limb
	/// \tparam rmask	mask which cancel out unwanted bits on the last
	/// \return hamming weight
	template<const uint32_t ulimb,const LimbType rmask>
	static uint32_t add_only_upper_weight_partly(LimbType *v3, LimbType const *v1, LimbType const *v2) noexcept {
		int32_t i = 0;
		uint32_t hm = 0;

// AVX optimisation disable, because i dont know a good way to calculate the hamming weight of one
//#ifdef USE_AVX2
//		#ifdef USE_AVX2_SPECIAL_ALIGNMENT
//		LOOP_UNROLL()
//		for (; i < ulimb/4; i++) {
//			__m256i x_avx = _mm256_load_si256((__m256i *)v1 + i);
//			__m256i y_avx = _mm256_load_si256((__m256i *)v2 + i);
//			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
//			_mm256_store_si256((__m256i *)v3.__data.data() + i, z_avx);
//		}
//
//		i*=4;
//#else
//        LOOP_UNROLL()
//        for (; i+4 < ulimb; i+=4) {
//            // we need to access the memory unaligned
//            __m256 x_avx = MM256_LOAD((float*)v1 + 2*i);
//            __m256 y_avx = MM256_LOAD((float*)v2 + 2*i);
//            __m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
//            MM256_STORE((float*)v3 + 2*i, z_avx);
//        }
//#endif
//#endif

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3[i] = v1[i] ^ v2[i];
			hm += popcount(v3[i]);
		}

		v3[ulimb] = v1[ulimb] ^ v2[ulimb];
		return hm + popcount(v3[ulimb] & rmask);
	}

	/// calculates the sumfof v3= v1+v2 on the partly length
	/// \tparam ulimb	max limb to calculate the weight on the full length of each limb
	/// \tparam rmask	mask which cancel out unwanted bits on the last
    template<const uint32_t ulimb, const LimbType rmask>
    constexpr static void add_only_upper(LimbType *v3, LimbType const *v1, LimbType const *v2) noexcept {
        if constexpr (0 == ulimb) {
            v3[0] = (v1[0] ^ v2[0]) & rmask;
            return;
        }

        int32_t i = 0;

#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		LOOP_UNROLL()
		for (; i < ulimb/4; i++) {
			__m256i x_avx = _mm256_load_si256((__m256i *)v1 + i);
			__m256i y_avx = _mm256_load_si256((__m256i *)v2 + i);
			__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
			_mm256_store_si256((__m256i *)v3 + i, z_avx);
		}

		i*=4;
#else
        LOOP_UNROLL()
        for (; i+4 < ulimb; i+=4) {
            // we need to access the memory unaligned
            __m256 x_avx = MM256_LOAD((float*)v1 + 2*i);
            __m256 y_avx = MM256_LOAD((float*)v2 + 2*i);
            __m256 z_avx = _mm256_xor_ps(x_avx, y_avx);
            MM256_STORE((float*)v3 + 2*i, z_avx);
        }
#endif
#endif

        LOOP_UNROLL();
        for (; i < ulimb; ++i) {
            v3[i] = v1[i] ^ v2[i];
        }

        v3[ulimb] = (v1[i] ^ v2[i]) & rmask;
    }

	constexpr static void add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                         const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		const LimbType lmask        = higher_mask(k_lower%limb_bits_width());
		const LimbType rmask        = lower_mask2(k_upper%limb_bits_width());
		const int64_t lower_limb    = k_lower / limb_bits_width();
		const int64_t higher_limb   = (k_upper-1) / limb_bits_width();

		if (lower_limb == higher_limb) {
			const LimbType mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			LimbType tmp1 = (v3.__data[lower_limb] & ~(mask));
			LimbType tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			return;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb+1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		LimbType tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		LimbType tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		LimbType tmp11 = (v3.__data[lower_limb] & ~(lmask));
		LimbType tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1^tmp11;
		v3.__data[higher_limb]= tmp2^tmp21;
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr static uint32_t add_weight(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		static_assert(k_upper <= length && k_lower < k_upper && 0 < k_upper);

		uint32_t cnorm = 0;
		constexpr LimbType lmask        = higher_mask(k_lower%limb_bits_width());
		constexpr LimbType rmask        = lower_mask2(k_upper%limb_bits_width());
		constexpr uint32_t lower_limb   = k_lower/limb_bits_width();
		constexpr uint32_t higher_limb  = (k_upper-1)/limb_bits_width();

		if constexpr (lower_limb == higher_limb) {
			constexpr LimbType mask = k_upper%limb_bits_width() == 0 ? lmask : (lmask & rmask);
			LimbType tmp1 = (v3.__data[lower_limb] & ~(mask));
			LimbType tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			auto b = popcount(tmp2);
			return b;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb+1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcount(v3.__data[i]);
		}

		LimbType tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		LimbType tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		LimbType tmp11 = (v3.__data[lower_limb] & ~(lmask));
		LimbType tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1^tmp11;
		v3.__data[higher_limb]= tmp2^tmp21;

		cnorm += popcount(tmp1);
		cnorm += popcount(tmp2);

		return cnorm;
	}

	constexpr static uint32_t add_weight(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                      const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper && 0 < k_upper);

		uint32_t cnorm = 0;
		const LimbType lmask        = higher_mask(k_lower%limb_bits_width());
		const LimbType rmask        = lower_mask2(k_upper%limb_bits_width());
		const int64_t lower_limb    = k_lower/limb_bits_width();
		const int64_t higher_limb   = (k_upper-1)/limb_bits_width();

		if (lower_limb == higher_limb) {
			const LimbType mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			LimbType tmp1 = (v3.__data[lower_limb] & ~(mask));
			LimbType tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			auto b = popcount(tmp2);
			return b;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb+1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcount(v3.__data[i]);
		}

		LimbType tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		LimbType tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		LimbType tmp11 = (v3.__data[lower_limb] & ~(lmask));
		LimbType tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1^tmp11;
		v3.__data[higher_limb]= tmp2^tmp21;

		cnorm += popcount(tmp1);
		cnorm += popcount(tmp2);

		return cnorm;
	}

	inline constexpr static bool add(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper,
	                       const uint32_t norm) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);

		if (norm == uint32_t(-1)){
			// fallback to normal addition.
			add(v3, v1, v2, k_lower, k_upper);
			return false;
		} else {
			ASSERT((&v1 != &v3 && &v2 != &v3) || (norm == -1));
			uint32_t cnorm = add_weight(v3, v1, v2, k_lower, k_upper);
			if (cnorm >= norm)
				return true;
		}

		return false;
	}

	// calculates the sum v3=v1+v2 and returns the hamming weight of v3
	inline constexpr static uint32_t add_weight(LimbType *v3, LimbType const *v1, LimbType const *v2) noexcept {
		uint32_t cnorm = 0;
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3[i] = v1[i] ^ v2[i];
			cnorm += popcount(v3[i]);
		}

		return cnorm;
	}

	inline constexpr static uint32_t add_weight(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		uint32_t cnorm = 0;
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcount(v3.__data[i]);
		}

		return cnorm;
	}


	// full length subtraction=addition in F_2
	inline int sub(BinaryContainer const &v) noexcept {
		return this->add(v);
	}

	/// alias for add
	inline void sub(BinaryContainer const &v, const uint32_t k_lower, const uint32_t k_upper) noexcept {
		return add(v, k_lower, k_upper);
	}

	/// alias for add
	inline constexpr static int sub(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		return add(v3, v1, v2);
	}

	/// alias for add
	inline constexpr static bool sub(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper) noexcept {
		add(v3, v1, v2, k_lower, k_upper);
		return false;
		//return add(v3, v1, v2, k_lower, k_upper);
	}

	/// alias for add
	inline static bool sub(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper,
	                       const uint32_t norm) noexcept {
		return add(v3, v1, v2, k_lower, k_upper, norm);
	}

	inline constexpr static bool cmp(BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		return cmp(v1, v2, 0, size());
	}

	/// implements only a 2 way comparison. E.g. implements the `!=` operator.
	inline constexpr static bool cmp(BinaryContainer const &v1, BinaryContainer const &v2,
	                       const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		const int32_t lower = round_down_to_limb(k_lower);
		const int32_t upper = round_down_to_limb(k_upper-1);
		const LimbType lmask = higher_mask(k_lower);
		const LimbType rmask = lower_mask2(k_upper);

		if (lower == upper) {   // the two offsets lay in the same limb.
			const LimbType mask = k_upper%limb_bits_width() == 0 ? lmask : (lmask & rmask);
			return cmp_simple2(v1, v2, lower, mask);
		} else {                // the two offsets lay in two different limbs
			// first check the highest limb with the mask
			return cmp_ext2(v1, v2, lower, upper, lmask, rmask);
		}
	}

	/// Important: lower != higher
	/// unrolled high speed implementation of a multi limb compare function
	template<const uint32_t lower, const uint32_t upper, const LimbType lmask, const LimbType umask>
	inline constexpr static bool cmp_ext(BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		ASSERT(lower != upper && lower < upper);

		// first check the highest limb with the mask
		if ((v1.__data[upper]&umask) != (v2.__data[upper]&umask))
			return false;

		// check all limbs in the middle
		LOOP_UNROLL()
		for(uint64_t i = upper-1; i > lower; i--) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		if ((v1.__data[lower]&lmask) != (v2.__data[lower]&lmask))
			return false;
		return true;
	}

	/// IMPORTANT; lower < upper so you have to compare at least two limbs.
	/// use this function if you have to compare a lot of different elements on the same coordinate. So you can precompute
	/// the mask and the limbs.
	inline constexpr static bool cmp_ext2(BinaryContainer const &v1, BinaryContainer const &v2,
	                           const uint32_t lower, const uint32_t upper, const LimbType lmask, const LimbType umask) noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// first check the highest limb with the mask.
		if ((v1.__data[upper]&umask) != (v2.__data[upper]&umask))
			return false;

		// check all limbs in the middle.
		for(uint64_t i = upper-1; i > lower; i--) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		// and at the end check the lowest limb.
		if ((v1.__data[lower]&lmask) != (v2.__data[lower]&lmask))
			return false;
		return true;
	}

	/// IMPORTANT: lower != higher => mask != 0. This is actually only a sanity check.
	/// high speed implementation of a same limb cmompare function
	template<const uint32_t limb, const LimbType mask>
	inline constexpr static bool cmp_simple(BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		ASSERT(limb != uint64_t(-1) && mask != 0);
		return ((v1.__data[limb]&mask) == (v2.__data[limb]&mask));
	}

	/// IMPORTANT: mask != 0.
	/// use this function if you have to compare a lot of different elements on the same coordinate. So you can precompute
	/// the mask and the limb.
	inline constexpr static bool cmp_simple2(BinaryContainer const &v1, BinaryContainer const &v2, const uint32_t limb, const LimbType mask) noexcept {
		ASSERT(limb != uint32_t(-1) && mask != 0);
		return ((v1.__data[limb]&mask) == (v2.__data[limb]&mask));
	}

	inline constexpr static int cmp_ternary_simple2(BinaryContainer const &v1, BinaryContainer const &v2, const uint32_t limb, const LimbType mask) noexcept {
		ASSERT(limb != uint64_t(-1));
		if ((v1.__data[limb]&mask) > (v2.__data[limb]&mask))
			return 1;
		else if ((v1.__data[limb]&mask) < (v2.__data[limb]&mask))
			return -1;

		return 0;
	}

	/// IMPORTANT: k_lower < k_upper is enforced.
	/// sets v1 = v2[k_lower, ..., k_upper].
	/// Does not change anything else.
	inline constexpr static void set(BinaryContainer &v1, BinaryContainer const &v2,
	                                 const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		const int64_t lower = round_down_to_limb(k_lower);
		const int64_t upper = round_down_to_limb(k_upper-1);
		const LimbType lmask = higher_mask(k_lower);
		const LimbType rmask = lower_mask2(k_upper);

		if (lower == upper) {   // the two offsets lay in the same limb.
			const LimbType mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			v1.__data[lower] = (v1.__data[lower] & ~mask) | (v2.__data[lower] & mask);
			return;
		} else {                // the two offsets lay in two different limbs
			v1.__data[upper] = (v1.__data[upper] & ~rmask) | (v2.__data[upper] & rmask);
			v1.__data[lower] = (v1.__data[lower] & ~lmask) | (v2.__data[lower] & lmask);
			for(uint64_t i = upper-1; i > lower; i--) {
				v1.data()[i] = v2.data()[i];
			}
		}
	}

	///  out[s: ] = in[0:s]
	static inline void shift_right(BinaryContainer &out, const BinaryContainer &in, const uint32_t s) noexcept {
		for (int j = 0; j < s; ++j) {
			out.write_bit(j+s, in.get_bit_shifted(j));
		}
	}

	/// checks whether this == obj on the interval [k_lower, ..., k_upper]
	/// the level of the calling 'list' object.
	/// \return
	inline bool is_equal(const BinaryContainer &obj, const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_equal(const BinaryContainer &obj) const noexcept {
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper-1);
		constexpr LimbType lmask = higher_mask(k_lower);
		constexpr LimbType rmask = lower_mask2(k_upper);

		if constexpr(lower == upper) {
			constexpr LimbType mask = lmask & rmask;
			return (__data[lower]^mask) == (obj.__data[lower]^mask);
		}

		return cmp_ext<lower, upper, lmask, rmask>(*this, obj);
	}

	template<const uint32_t lower, const uint32_t upper, const LimbType lmask, const LimbType umask>
	inline bool is_equal_ext(const BinaryContainer &obj) const noexcept {
		return cmp_ext2<lower, upper, lmask, umask>(*this, obj);
	}

	inline bool is_equal_ext2(const BinaryContainer &obj, const uint32_t lower, const uint32_t upper,
						                                  const LimbType lmask, const LimbType umask) const noexcept {
		return cmp_ext2(*this, obj, lower, upper, lmask, umask);
	}

	inline bool is_equal_simple2(const BinaryContainer &obj, const uint32_t limb, const LimbType mask) const noexcept {
		return cmp_simple2(*this, obj, limb, mask);
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_greater(BinaryContainer const &obj) const noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper-1);
		constexpr LimbType lmask = higher_mask(k_lower);
		constexpr LimbType rmask = lower_mask2(k_upper);
		if constexpr (lower == upper) {   // the two offsets lay in the same limb.
			constexpr LimbType mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			return ((__data[lower]&mask) > (obj.__data[lower]&mask));
		} else {                // the two offsets lay in two different limbs
			ASSERT(0);
			return 0;
		}
	}

	/// implements a strict comparison. Call this function if you dont know what to call. Its the most generic implementaion
	/// and it works for all input.s
	inline bool is_greater(BinaryContainer const &obj, const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		int64_t lower = round_down_to_limb(k_lower);
		int64_t upper = round_down_to_limb(k_upper-1);
		const LimbType lmask = higher_mask(k_lower);
		const LimbType rmask = lower_mask2(k_upper);

		if (lower == upper) {   // the two offsets lay in the same limb.
			const LimbType mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			return this->is_greater_simple2(obj, lower, mask);
		} else {                // the two offsets lay in two different limbs
			return this->is_greater_ext2(obj, lower, upper, lmask, rmask);
		}
	}

	/// return *this > obj on the limbs [lower, upper]. `lmask` and `umask` are bitmask for the lowest and highest limbs.
	/// Technically this is a specially unrolled implementation of `BinaryContainer::is_greater` if you have to compare a lot
	/// of containers repeatedly on the same coordinates.
	inline bool is_greater_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                          const LimbType lmask, const LimbType umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldn't make sense.

		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, >, <)
		// check all limbs in the middle
		for(uint64_t i = upper-1; i > lower; i--) {
			BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], >, <)
		}

		BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, >, <)
		return false;
	}

	inline bool is_greater_equal_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                            const LimbType lmask, const LimbType umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldn't make sense.
		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, >=, <)
		// check all limbs in the middle
		for(uint64_t i = upper-1; i > lower; i--) {
			BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], >=, <)
		}

		BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, >=, <)
		return false;
	}

	/// most simple type of comparison implemented for this class.
	/// returns *this < obj on bits specified by the parameter `limb` and `mask.`
	/// call like this:
	///		using BinaryContainerTest = BinaryContainer<64>;
	///		BinaryContainerTest b1, b2;
	///		uint64_t limb = 0;
	///		mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
	///		b1.is_greater_simple2(b2, limb, mask);
	inline bool is_greater_simple2(BinaryContainer const &obj, const uint32_t limb, const LimbType mask) const noexcept {
		ASSERT(limb < limbs() && mask != 0);
		return ((__data[limb]&mask) > (obj.__data[limb]&mask));
	}

	/// not testet
	inline bool is_greater_equal_simple2(BinaryContainer const &obj, const uint32_t limb, const LimbType mask) const noexcept {
		ASSERT(limb < limbs() && mask != 0);
		return ((__data[limb]&mask) >= (obj.__data[limb]&mask));
	}

	/// main comparison function for the < operator. If you dont know what function to use, use this one. Its the most generic
	/// implementation and works for all inputs.
	inline bool is_lower(BinaryContainer const &obj, const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		int64_t lower = round_down_to_limb(k_lower);
		int64_t upper = round_down_to_limb(k_upper-1);
		const LimbType lmask = higher_mask(k_lower);
		const LimbType rmask = lower_mask2(k_upper);

		if (lower == upper) {   // the two offsets lay in the same limb.
			const LimbType mask = k_upper%64 == 0 ? lmask : (lmask & rmask);
			return is_lower_simple2(obj, lower, mask);
		} else {                // the two offsets lay in two different limbs
			return is_lower_ext2(obj, lower, upper, lmask, rmask);
		}
	}

	/// return *this < obj on the limbs [lower, upper]. `lmask` and `umask` are bitmask for the lowest and highest limbs.
	/// Technically this is a specially unrolled implementation of `BinaryContainer::is_lower` if you have to compare a lot
	/// of containers repeatedly on the same coordinates.
	/// Example Code:
	///		using BinaryContainerTest = BinaryContainer<G_n>;
	///
	///		BinaryContainerTest2 b1, b2;
	///			FILL WITH DATA HERE.
	///		const uint64_t lower = BinaryContainerTest2::round_down_to_limb(k_higher);
	///		const uint64_t upper = BinaryContainerTest2::round_down_to_limb(b1.size()-1);
	///		const BinaryContainerTest2::LimbType lmask = BinaryContainerTest2::higher_mask(k_higher);
	///		const BinaryContainerTest2::LimbType umask = BinaryContainerTest2::lower_mask2(b1.size());
	///		is_lower_ext2(b2, lower, upper, lmask, umask))
	/// You __MUST__ be extremely carefully with the chose of `upper`.
	/// Normally you want to compare two elements between `k_lower` and `k_upper`. This can be done by:
	///		const uint64_t lower = BinaryContainerTest2::round_down_to_limb(k_lower);
	///		const uint64_t upper = BinaryContainerTest2::round_down_to_limb(k_higher);
	///							... (as above)
	/// Note that you dont have to pass the -1 to k_higher. The rule of thumb is that you __MUST__ add a -1 to the computation
	/// of the upper limb.
	inline bool is_lower_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
						                                  const LimbType lmask, const LimbType umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldnt make sense.

		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, <, >)
		// check all limbs in the middle
		for(uint64_t i = upper-1; i > lower; i--) {
			BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], <, >)
		}

		BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, <, >)
		return false;
	}

	/// not testet
	inline bool is_lower_equal_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                          const LimbType lmask, const LimbType umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldnt make sense.

		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, <=, >)
		// check all limbs in the middle
		for(uint64_t i = upper-1; i > lower; i--) {
			BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], <=, >)
		}

		BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, <=, >)
		return false;
	}

	// efficient reimplementation of `is_lower` for the special case that `k_lower` and `k_upper` are in the same limb.
	/// call like this:
	///		using BinaryContainerTest = BinaryContainer<64>;
	///		BinaryContainerTest b1, b2;
	///			FILL WITH DATA
	///		uint64_t limb = 0;
	///		mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
	///		b1.is_lower_simple2(b2, limb, mask);
	inline bool is_lower_simple2(BinaryContainer const &obj, const uint32_t limb, const LimbType mask) const noexcept {
		ASSERT(limb < limbs() < length && mask != 0);
		return ((__data[limb]&mask) < (obj.__data[limb]&mask));
	}

	inline bool is_lower_equal_simple2(BinaryContainer const &obj, const uint32_t limb, const LimbType mask) const noexcept {
		ASSERT(limb < limbs() < length && mask != 0);
		return ((__data[limb]&mask) <= (obj.__data[limb]&mask));
	}

	// calculates the weight = hamming weight of the data container.
	inline uint32_t weight() const noexcept {
		return weight(0, length);
	}

	// calcs the weight up to (include) ilumb at early exits if its bigger than early exit.
	template<const uint32_t ulimb,const LimbType rmask, const uint32_t early_exit>
	inline uint32_t weight_earlyexit() noexcept {
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (uint32_t i = 0; i < ulimb; ++i) {
			hm += popcount(__data[i]);
			if (hm > early_exit)
				return hm;
		}

		return hm + popcount(__data[ulimb] & rmask);
	}

	uint32_t weight(const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_upper <= length && k_lower < k_upper);
		const uint32_t lower = round_down_to_limb(k_lower);
		const uint32_t upper = round_down_to_limb(k_upper-1);

		const LimbType l_mask = higher_mask(k_lower);
		const LimbType u_mask = lower_mask2(k_upper);

		uint32_t weight;
		// if only one limb needs to be checked to check
		if(lower == upper){
			uint64_t b = (l_mask & u_mask);
			uint64_t c = uint64_t(__data[lower]);
			uint64_t d = uint64_t(b) & uint64_t(c);
			uint64_t w_ = popcount(d);
			return w_;
		}

		weight = popcount(l_mask&__data[lower]);
		weight += popcount(u_mask&__data[upper]);
		for(uint32_t i=lower+1;i<upper;++i)
			weight += popcount(__data[i]);

		return weight;
	}

	template<const uint32_t lower, const uint32_t upper, const LimbType l_mask, const LimbType u_mask>
	constexpr uint32_t weight() const noexcept {
		ASSERT(lower <= upper);
		uint32_t weight = 0;

		// if only one limb needs to be checked to check
		if constexpr (lower == upper){
			uint64_t b = (l_mask & u_mask);
			uint64_t c = uint64_t(__data[lower]);
			uint64_t d = uint64_t(b) & uint64_t(c);
			uint64_t w_ = popcount(d);
			return w_;
		}

		weight = popcount(l_mask&__data[lower]);
		for(uint32_t i=lower+1;i<upper;++i)
			weight += popcount(__data[i]);
		weight += popcount(u_mask&__data[upper]);
		return weight;
	}

	/// Calculates a+=b on full length while caluclating the hamming wight only on the first coordinates, which are
	/// specified by `upper` and `u_mask`
	/// \tparam upper	limb position of the last normal addition
	/// \tparam u_mask	mask to apply the calculate the weight on the last limb
	/// \param a	input vector
	/// \param b	intpur vector
	/// \return	hamming weight of the first coordinates.
	template<const uint32_t upper, const LimbType u_mask>
	constexpr static uint32_t weight_sum_only_upper(LimbType *a, LimbType *b) noexcept {
		uint32_t weight = 0;

		// if only one limb needs to be checked to check
		if constexpr (0 == upper){
			LimbType c = LimbType(a[0] ^ b[0]);
			LimbType d = LimbType(u_mask) & LimbType(c);
			LimbType w_ = popcount(d);
			return w_;
		}

		for(uint32_t i = 0; i < upper;++i)
			weight += popcount(a[i] ^ b[i]);

		return weight + popcount(u_mask & (a[upper] ^ b[upper]));
	}

	// hack it like
	class reference {
		friend class BinaryContainer;

		// pointer to the limb
		LimbType     *wp;
		// bit position in the whole data array.
		const size_t 	        mask_pos;

		// left undefined
		reference();

	public:
		reference(BinaryContainer &b, size_t pos) : mask_pos(mask(pos)){
			wp = &b.data().data()[round_down_to_limb(pos)];
		}

#if __cplusplus >= 201103L
		reference(const reference&) = default;
#endif

		~reference() = default;

		// For b[i] = __x;
		reference& operator=(bool x) {
			if (x)
				*wp |= mask_pos;
			else
				*wp &= ~mask_pos;
			return *this;
		}

		// For b[i] = b[__j];
		reference& operator=(const reference& j) {
			if (*(j.wp) & j.mask_pos)
				*wp |= mask_pos;
			else
				*wp &= ~mask_pos;
			return *this;
		}

		// Flips the bit
		bool operator~() const { return (*(wp) & mask_pos) == 0; }

		// For __x = b[i];
		operator bool() const {
			return (*(wp) & mask_pos) != 0;
		}

		// For b[i].flip();
		reference& flip() {
			*wp ^= mask_pos;
			return *this;
		}

		unsigned int get_data() const { return bool(); }
		unsigned int data() const { return bool(); }
	};
	friend class reference;


	reference operator[](size_t pos) noexcept { return reference(*this, pos); }
	constexpr bool operator[](const size_t pos) const noexcept { return (__data[round_down_to_limb(pos)] & mask(pos)) != 0; }

	/// Assignment operator implementing copy assignment
	/// see https://en.cppreference.com/w/cpp/language/operators
	/// \param obj
	/// \return
	BinaryContainer& operator =(BinaryContainer const &obj) noexcept {
		if (this != &obj) { // self-assignment check expected
			std::copy(&obj.__data[0], &obj.__data[0] + obj.__data.size(), &this->__data[0]);
		}

		return *this;
	}

	/// Assignment operator implementing move assignment
	/// Alternative definition: Value& operator =(Value &&obj) = default;
	/// see https://en.cppreference.com/w/cpp/language/move_assignment
	/// \param obj
	/// \return
	BinaryContainer& operator =(BinaryContainer &&obj) noexcept {
		if (this != &obj) { // self-assignment check expected really?
			// move the data
			__data = std::move(obj.__data);
		}

		return *this;
	}

	/// print some information
	/// \param k_lower lower limit to print (included)
	/// \param k_upper higher limit to print (not included)
	void print(const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_lower < length && k_upper <= length && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << data(i) << "";
		}
		std::cout << "\n";
	}

	//LimbType data(uint64_t index) { ASSERT(index < length); return get_bit_shifted(index); }
	bool data(uint64_t index) const noexcept { ASSERT(index < length); return get_bit_shifted(index); }

	LimbType get_type() {return __data[0]; }

#ifdef BINARY_CONTAINER_ALIGNMENT
	static constexpr uint16_t alignment() {
		// Aligns to a multiple of 32 Bytes
		//constexpr uint16_t bytes = (length+7)/8;
		//constexpr uint16_t limbs = (((bytes-1) >> 5)+1)<<5;
		//return limbs*8;

		// Aligns to a multiple of 16 Bytes
		constexpr uint16_t bytes = (length+7)/8;
		constexpr uint16_t limbs = (((bytes-1) >> 4)+1)<<4;
		return limbs*8;
	}
#endif

	// length operators
	__FORCEINLINE__ constexpr static bool binary() noexcept { return true; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return length; }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return (length+limb_bits_width()-1)/limb_bits_width(); }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept {
#ifdef BINARY_CONTAINER_ALIGNMENT
		return alignment()/8;
#else
		return limbs()*sizeof(LimbType);
#endif
	}

	__FORCEINLINE__ LimbType* ptr() noexcept { return __data.data(); };
	const __FORCEINLINE__ LimbType* ptr() const noexcept { return __data.data(); };

	__FORCEINLINE__ std::array<LimbType, compute_limbs()>& data() noexcept { return __data; };
	__FORCEINLINE__ const std::array<LimbType, compute_limbs()>& data() const noexcept { return __data; };

private:
	// actual data container.
	std::array<LimbType, compute_limbs()> __data;
};

// https://www.cs.purdue.edu/homes/xyzhang/fall14/lock_free_set.pdf
// https://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
// https://users.fmi.uni-jena.de/~nwk/LockFree.pdf
// IMPORTANT: T must implement a field next
template<typename T, int (*c)(const T*, const T*)>
class ConstNonBlockingLinkedList {
private:
	// Atomic Type
	using AT = std::atomic<T *>;
public:
	ConstNonBlockingLinkedList() {
		head.store(nullptr);
	}

	/// IDEA: preallocate len elements?
	explicit ConstNonBlockingLinkedList(const uint32_t len) {
		// make sure that the head is correctly initialised
		head.store(nullptr);
	}

	// only needed for the C version.
	int CAS(void **mem, void *o, void *n) {
		int res;
		asm("lock cmpxchg %3,%1; "
			"mov $0,%0;"
			"jnz 1f; "
			"inc %0; 1:"
			: "=a" (res) : "m" (*mem), "a" (o), "d" (n));
		return res;
	}


	/// insert front, unsorted
	void insert(T *a) {
		auto newhead = head.load();
		do {
			a->next.store(newhead);
		} while (!head.compare_exchange_weak(newhead, a, std::memory_order_release, std::memory_order_relaxed));
	}

	/// insert back, unsorted
	void insert_back(T *a) {
		auto back = traverse();
		auto newback = back->load();
		back->compare_exchange_strong(newback, a, std::memory_order_release, std::memory_order_relaxed);
	}

	// return the last element in the list.
	AT* traverse() {
		// catch the case where the list is empty
		if (head.load() == nullptr)
			return &head;

		auto newhead = &head;
		while (newhead->load()->next != nullptr){
			newhead = &newhead->load()->next;
		}

		return newhead;
	}

	/// returns the middle node of the linked list.
	T* middle_node() {
		T *first = head.load();
		T *last = traverse()->load();

		if (first == nullptr) {
			return nullptr;
		}

		T *sm = first;
		T *fm = first->next;
		while (fm != last) {
			fm = fm->next;
			if (fm != last) {
				sm = sm->next;
				fm = fm->next;
			}
		}

		return sm;
	}

	T* binary_search(const T *p) {
		T *fn= head.load();
		T *ln = nullptr;
		T *cn = nullptr;
		int tmp;

		if (fn == nullptr) {
			return nullptr;
		}

		cn = middle_node(fn, ln);
		while ((cn != nullptr) && (fn != ln)) {
			tmp = c(cn, p);
			if (tmp == 0) {
				return cn;
			} else if (tmp == -1) {
				fn = cn->next;
			} else {
				ln = cn;
			}

			if (fn != ln) {
				cn = middle_node(fn, ln);
			}
		}

		return nullptr;
	}

	void insert_sorted(const T *a) {
		T *fn= head.load();
		T *ln = nullptr;
		T *cn = nullptr;

		if (fn == nullptr) {
			// input list empty
			insert(a);
			return;
		} else if (c(fn->next, a) >= 1) {
			insert(a);
			return;
		}

		cn = middle_node(fn, ln);
		while ((cn->next != nullptr)  && (fn != ln)) {
			if (c(cn->next, a) == -1) {
				fn = cn->next;
			} else {
				ln = cn;
			}

			if (fn != ln) {
				cn = middle_node(fn, ln);
			}
		}

		a->next = fn->next;
		fn->next = a;
	}


	void print(){
		auto next = head.load();
		while (next != nullptr) {
			std::cout << next->data << "\n";
			next = next->next.load();
		}
	}
	AT head;
};


template<typename T, uint32_t length>
std::ostream& operator<< (std::ostream &out, const kAryContainer_T<T, length> &obj) {
	for (uint64_t i = 0; i < obj.get_size(); ++i) {
		out << obj[i] << " ";
	}
	return out;
}

template<class T, const T MOD, uint32_t length>
std::ostream& operator<< (std::ostream &out, const kAryPackedContainer_T<T, MOD, length> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << unsigned(obj[i]);
	}
	return out;
}

template<uint32_t length>
std::ostream& operator<< (std::ostream &out, const BinaryContainer<length> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << obj[i];
	}
	return out;
}

#endif //SMALLSECRETLWE_CONTAINER_H
