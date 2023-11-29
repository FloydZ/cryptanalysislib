#ifndef SMALLSECRETLWE_FQ_PACKED_VECTOR_H
#define SMALLSECRETLWE_FQ_PACKED_VECTOR_H

#include <array>
#include <cstdint>
#include <atomic>
#include <cstddef>

#include "helper.h"
#include "random.h"
#include "popcount/popcount.h"
#include "simd/simd.h"
#include "container/binary_packed_vector.h"


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
/// \param n = number of elemtns
/// \param q = modulus
template<class T, const uint32_t n, const uint32_t q>
#if __cplusplus > 201709L
	requires std::is_integral<T>::value
#endif
class kAryPackedContainer_Meta {
	public:

	typedef kAryPackedContainer_Meta ContainerType;

	// number of bits in each T
	constexpr static uint16_t bits_per_limb = sizeof(T)*8;
	// number of bits needed to represent MOD
	constexpr static uint16_t bits_per_number = (uint16_t)constexpr_bits_log2(q);
	// number of numbers one can fit into each limb
	constexpr static uint16_t numbers_per_limb = (sizeof(T)*8u) / bits_per_number;
	// Number of Limbs needed to represent `length` numbers of size log(MOD) +1
	constexpr static uint16_t internal_limbs = std::max(1u, (n+numbers_per_limb-1)/ numbers_per_limb);
	// mask with the first `bits_per_number` set to one
	constexpr static T number_mask = (T(1u) << (bits_per_number)) - 1u;

	// true if we need every bit of the last bit
	constexpr static bool is_full = (n%bits_per_limb) == 0;

	constexpr static bool activate_avx2 = true;

	// we are good C++ devs.
	typedef T ContainerLimbType;
	using DataType = LogTypeTemplate<bits_per_number>;

	// list compatibility typedef
	typedef T LimbType;
	typedef T LabelContainerType;

	// make the length and modulus of the container public available
	constexpr static uint32_t LENGTH = n;
	constexpr static uint32_t MODULUS = q;

	// internal data
	std::array<T, internal_limbs> __data;

	/// the mask is only valid for one internal number.
	/// \param i bit position the read
	/// \return bit mask to access the i-th element within a limb
	constexpr inline T accessMask(const uint16_t i) const noexcept {
		return number_mask << (i%numbers_per_limb);
	}

	/// access the i-th coordinate/number/element
	/// \param i coordinate to access.
	/// \return the number you wanted to access, shifted down to the lowest bits.
	constexpr DataType get(const uint16_t i) const noexcept {
		// needs 5 instructions. So 64*5 for the whole limb
		ASSERT(i < LENGTH);
		return DataType((__data[i/numbers_per_limb] >> ((i%numbers_per_limb)*bits_per_number)) & number_mask);
	}

	/// sets the `i`-th number to `data`
	/// \param data value to set the array on
	/// \param i -th number to overwrite
	/// \retutn nothing
	constexpr void set(const DataType data, const uint32_t i) noexcept {
		ASSERT(i < LENGTH);
		const uint16_t off = i/numbers_per_limb;
		const uint16_t spot = (i%numbers_per_limb) * bits_per_number;

		T bla = (number_mask&T(data%MODULUS)) << spot;
		T val = __data[off] & (~(number_mask << spot));
		__data[off] = bla | val;
	}

	/// sets the `i`-th number to `data`
	/// \param data value to set the array on
	/// \param i -th number to overwrite
	/// \retutn nothing
	constexpr void set(const DataType data) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < n; ++i) {
			set(data, i);
		}
	}

	/// set everything to zero
	constexpr void zero() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < internal_limbs; i++){
			__data[i] = 0;
		}
	}

	/// sets everything to zero between [a, b)
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void zero(const uint32_t a, const uint32_t b) noexcept {
		for (uint32_t i = a; i < b; i++){
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
	constexpr void one(const uint32_t a=0, const uint32_t b=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(1, i);
		}
	}

	/// Set everything to two
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void two(const uint32_t a=0, const uint32_t b=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i = a; i < b; i++) {
			set(2%MODULUS, i);
		}
	}

	/// generates all limbs uniformly random
	/// \param a lower bound, inclusive
	/// \param b higher bound, exclusive
	/// \return nothing
	constexpr void random(const uint32_t a=0, const uint32_t b=LENGTH) noexcept {
		LOOP_UNROLL();
		for (uint32_t i=a; i < b; i++) {
			set(fastrandombytes_uint64() % MODULUS, i);
		}
	}

	/// computes the hamming weight
	[[nodiscard]] constexpr uint32_t popcount() const noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < size(); i++) {
			ret += get(i) > 0;
		}

		return ret;
	}

	/// return the positions of the first p bits/numbers set
	/// \param out output: array of the first p positions set in the container
	/// \param p maximum bits!=0 to find
	constexpr void get_bits_set(uint16_t *out, const uint32_t p) const noexcept {
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
	[[nodiscard]] constexpr bool is_zero() const noexcept {
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
	[[nodiscard]] constexpr bool is_zero(const uint32_t a, const uint32_t b) const noexcept {
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
	[[nodiscard]] constexpr inline uint32_t weight(const uint32_t a=0, const uint32_t b=LENGTH) const noexcept {
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
	constexpr void swap(const uint16_t i, const uint16_t j) noexcept {
		ASSERT(i < LENGTH && j < LENGTH);
		auto tmp = get(i);
		set(i, get(j));
		set(j, tmp);
	}


	/// infix negates (x= -x mod q)  all numbers between [k_lower, k_higher)
	/// \param k_lower lower limit, inclusive
	/// \param k_upper higher limit, exclusive
	constexpr inline void neg(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH &&k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			set(((-get(i)) + MODULUS)%MODULUS, i);
		}
	}

	/// negates (x = -q mod q) betweet [k_lower, k_upper)
	/// \tparam k_lower inclusive
	/// \tparam k_upper exclusive
	template<uint32_t k_lower, uint32_t k_upper>
	constexpr inline void neg() noexcept {
		for (uint32_t i = k_lower; i < k_upper; i++) {
			set(((-get(i)) + MODULUS)%MODULUS, i);
		}
	}

	/// TODO write test
	/// NOTE: generic implementation
	/// \tparam TT
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT>
	constexpr inline static TT add_T(const TT in1, const TT in2) {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT)*8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			const TT b = (in2 >> (bits_per_number * i)) & mask;
			ret ^= ((a+b)%q) << (bits_per_number * i);
		}
		return ret;
	}

	/// TODO write test
	template<typename TT>
	constexpr inline static TT sub_T(const TT in1, const TT in2) {
		static_assert(sizeof(TT) <= 16);
		constexpr uint32_t nr_limbs = (sizeof(TT)*8) / bits_per_number;
		constexpr TT mask = (1ull << bits_per_number) - 1ull;

		TT ret = 0;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			const TT a = (in1 >> (bits_per_number * i)) & mask;
			const TT b = (in2 >> (bits_per_number * i)) & mask;
			ret ^= ((a-b+q)%q) << (bits_per_number * i);
		}
		return ret;
	}

	/// v1 += v1
	/// \param v2 input/output
	/// \param v1 input
	constexpr inline static void add(kAryPackedContainer_Meta &v1,
									 kAryPackedContainer_Meta const &v2) noexcept {
		add(v1, v1, v2, 0, LENGTH);
	}

	/// v3 = v1 + v2
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	constexpr inline static void add(kAryPackedContainer_Meta &v3,
	                       			 kAryPackedContainer_Meta const &v1,
	                       			 kAryPackedContainer_Meta const &v2) noexcept {
		add(v3, v1, v2, 0, LENGTH);
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
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = v1.get(i) + v2.get(i);
			v3.set(data % MODULUS, i);
		}
	}

	/// v1 -= v2
	/// \param v1 input/output
	/// \param v2 input
	constexpr inline static void sub(kAryPackedContainer_Meta &v1,
									 kAryPackedContainer_Meta const &v2) noexcept {
		sub(v1, v1, v2, 0, LENGTH);
	}

	/// v3 = v1 - v2 mod q
	/// NOTE: this computes the subtraction on all coordinates
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	constexpr inline static void sub(kAryPackedContainer_Meta &v3,
	                       			 kAryPackedContainer_Meta const &v1,
	                       			 kAryPackedContainer_Meta const &v2) noexcept {
		sub(v3, v1, v2, 0, LENGTH);
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
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			int64_t data = int64_t(v1.get(i)) - int64_t(v2.get(i));
			if (data < 0)
				data += MODULUS;
			v3.set(data % MODULUS, i);
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
									 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = (v1.get(i) * v2.get(i)) % MODULUS;
			v3.set(data, i);
		}
	}

	/// generic components mul: v3 = v1 * v2 between [k_lower, k_upper)
	/// \param v3 output
	/// \param v1 input
	/// \param v2 input
	/// \param k_lower lower limit inclusive
	/// \param k_upper upper limit exclusive
	constexpr inline static void scalar(kAryPackedContainer_Meta &v3,
						   				kAryPackedContainer_Meta const &v1,
	                       				const DataType v2,
						   				const uint32_t k_lower=0,
	                                    const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			DataType data = (v1.get(i) * v2) % MODULUS;
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
	                       			 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
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
	                       			 const uint32_t k_lower=0,
	                                 const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			v1.set(v2.get(i), i);
		}
	}

	/// \param obj
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	/// \return this == obj between [k_lower, k_upper)
	constexpr bool is_equal(kAryPackedContainer_Meta const &obj,
	              			const uint32_t k_lower=0,
	                        const uint32_t k_upper=LENGTH) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	///
	/// \param obj
	/// \param k_lower inclusive
	/// \param k_upper exclusive
	/// \return this > obj [k_lower, k_upper)
	constexpr bool is_greater(kAryPackedContainer_Meta const &obj,
	                		  const uint32_t k_lower=0,
	                          const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_upper; i > k_lower; i++) {
			if (get(i-1) < obj.get(i-1))
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
	                        const uint32_t k_lower=0,
	                        const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			if(get(i) > obj.get(i))
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
		ASSERT(i < LENGTH);
		for (uint32_t j = LENGTH; j > i; j--) {
			set(get(j-i-1), j-1);
		}

		// clear the rest.
		for (uint32_t j = 0; j < i; j++) {
			set(0, j);
		}
	}

	/// shifts this by i to the left
	/// \param i amount to shift
	constexpr void right_shift(const uint32_t i) noexcept {
		ASSERT(i < LENGTH);
		for (uint32_t j = 0; j < n - i; j--) {
			const auto data = get(i + j);
			set(data, j);
		}

		// clear the rest.
		for (uint32_t j = 0; j < i; j++) {
			set(0, n-j-1u);
		}
	}

	///
	/// \param obj
	/// \return
	kAryPackedContainer_Meta& operator =(kAryPackedContainer_Meta const &obj) noexcept {
		ASSERT(size() == obj.size() && "äh?");

		if (likely(this != &obj)) {
			__data = obj.__data;
		}

		return *this;
	}

	/// prints between [k_lower, k_upper )
	/// \param k_lower lower limit
	/// \param k_upper upper limit
	void print(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < LENGTH && k_upper <= LENGTH && k_lower < k_upper);
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
	                         const uint16_t k_lower=0,
	                         const uint16_t k_higher=LENGTH) noexcept {
		uint64_t d = data;
		for (uint32_t i = k_lower; i < k_higher; ++i) {
			std::cout << unsigned(d&1);
			std::cout << unsigned((d>>1)&1);
			d >>= 2;
		}
		std::cout << "\n";
	}

	/// print the `pos
	/// \param k_lower inclusive
	/// \param k_higher exclusive
	void print_binary(const uint32_t k_lower=0, const uint32_t k_higher=LENGTH) const noexcept {
		for (uint32_t i = k_lower; i < k_higher; ++i) {
			std::cout << unsigned(get(i)&1);
			std::cout << unsigned((get(i)>>1)&1);
		}
		std::cout << "\n";
	}

	// iterators.
	auto begin() noexcept { return __data.begin();}
	auto begin() const noexcept { return __data.begin();}
	auto end() noexcept { return __data.end();}
	auto end() const noexcept { return __data.end();}

	///
	/// \param i
	/// \return
	DataType operator [](size_t i) noexcept {
		ASSERT(i < LENGTH );
		return get(i);
	}

	///
	/// \param i
	/// \return
	DataType operator [](const size_t i) const noexcept {
		ASSERT(i < LENGTH );
		return get(i);
	};

	// return `true` if the datastruct contains binary data.
	__FORCEINLINE__ constexpr static bool binary() noexcept { return false; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return LENGTH; }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return internal_limbs; }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return internal_limbs*sizeof(T); }

	std::array<T, internal_limbs>& data() noexcept { return __data; }
	const std::array<T, internal_limbs>& data() const noexcept { return __data; }

	/// TODO use the reference class trick as in binary_vector.h
	//T& data(const size_t index) { ASSERT(index < length && "wrong index"); return __data[index]; }
	const T data(const size_t index) const noexcept { ASSERT(index < LENGTH && "wrong index"); return __data[index]; }
	T limb(const size_t index) { ASSERT(index < LENGTH && "wrong index"); return __data[index]; }
	const T limb(const size_t index) const noexcept { ASSERT(index < LENGTH && "wrong index"); return __data[index]; }

	// get raw access to the underlying data.
	T* ptr() noexcept { return __data.data(); }
	const T* ptr() const noexcept { return __data.data(); }

	// returns `false` as this class implements a generic arithmetic
	__FORCEINLINE__  static constexpr bool optimized() noexcept { return false; };
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
class kAryPackedContainer_T : public kAryPackedContainer_Meta<T, n ,q> {
	/// Nomenclature:
	///     Number 	:= actual data one wants to save % MODULUS
	///		Limb 	:= Underlying data container holding at max sizeof(T)/log2(MODULUS) many numbers.
	/// Its not possible, that numbers cover more than one limb.
	/// The internal data container layout looks like this:
	/// 		limb0				limb1			limb2
	///   [	n0	,  n1  ,  n2  |	    , 	,	  |		,	 ,     |  .... ]
	/// The container fits as much numbers is the limb as possible. But there will no overhanging elements (e.g.
	///  numbers that first bits are on one limb and the remaining bits are on the next limb).
	///

public:
	using kAryPackedContainer_Meta<T, n ,q>::bits_per_limb;
	using kAryPackedContainer_Meta<T, n ,q>::bits_per_number;
	using kAryPackedContainer_Meta<T, n ,q>::numbers_per_limb;
	using kAryPackedContainer_Meta<T, n ,q>::internal_limbs;
	using kAryPackedContainer_Meta<T, n ,q>::number_mask;
	using kAryPackedContainer_Meta<T, n ,q>::is_full;
	using kAryPackedContainer_Meta<T, n ,q>::activate_avx2;



	using typename kAryPackedContainer_Meta<T, n ,q>::ContainerLimbType;
	using typename kAryPackedContainer_Meta<T, n ,q>::LimbType;
	using typename kAryPackedContainer_Meta<T, n ,q>::LabelContainerType;
	// minimal internal datatype to present an element.
	using DataType = LogTypeTemplate<bits_per_number>;

	using kAryPackedContainer_Meta<T, n ,q>::LENGTH;
	using kAryPackedContainer_Meta<T, n ,q>::MODULUS;
	using kAryPackedContainer_Meta<T, n ,q>::__data;
public:

	// base constructor
	constexpr kAryPackedContainer_T () = default;
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
class kAryPackedContainer_T<uint64_t , n, 3> : public kAryPackedContainer_Meta<uint64_t , n, 3> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;
	using T = uint64_t;

	/// needed size descriptions
	using kAryPackedContainer_Meta<T, n ,q>::bits_per_limb;
	using kAryPackedContainer_Meta<T, n ,q>::bits_per_number;
	using kAryPackedContainer_Meta<T, n ,q>::numbers_per_limb;
	using kAryPackedContainer_Meta<T, n ,q>::internal_limbs;
	using kAryPackedContainer_Meta<T, n ,q>::number_mask;
	using kAryPackedContainer_Meta<T, n ,q>::is_full;
	using kAryPackedContainer_Meta<T, n ,q>::activate_avx2;

	/// needed type definitions
	using typename kAryPackedContainer_Meta<T, n ,q>::ContainerLimbType;
	using typename kAryPackedContainer_Meta<T, n ,q>::LimbType;
	using typename kAryPackedContainer_Meta<T, n ,q>::LabelContainerType;
	// minimal internal datatype to present an element.
	using DataType = LogTypeTemplate<bits_per_number>;

	/// needed
	using kAryPackedContainer_Meta<T, n ,q>::LENGTH;
	using kAryPackedContainer_Meta<T, n ,q>::MODULUS;
	using kAryPackedContainer_Meta<T, n ,q>::__data;

	typedef kAryPackedContainer_T<T, n, q> ContainerType;

	/// some functions
	using kAryPackedContainer_Meta<T, n ,q>::is_equal;
	using kAryPackedContainer_Meta<T, n ,q>::is_greater;
	using kAryPackedContainer_Meta<T, n ,q>::is_lower;
	using kAryPackedContainer_Meta<T, n ,q>::add;
public:

	/// calculates the hamming weight of one limb.
	/// NOTE only correct if there is no 3 in one of the limbs
	/// \param a
	/// \return
	static inline uint16_t hammingweight_mod3_limb(const T a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr T c1 = T(6148914691236517205u);
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T c2 = T(12297829382473034410u);

		const T ac1 = a&c1; // filter the ones
		const T ac2 = a&c2; // filter the twos

		return __builtin_popcountll(ac1) + __builtin_popcountll(ac2);
	}

	/// calculates the hamming weight of one __uint128
	/// NOTE only correct if there is no 3 in one of the limbs
	/// \param a
	/// \return
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
	/// computes the hamming weight of a ternary vector
	/// \param a input
	/// \return a mod 3, in every coordinate
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

	constexpr inline void neg(const uint32_t lower, const uint32_t upper){
		ASSERT(false);
	}

	/// negates the vector on all coordinates
	constexpr inline void neg() noexcept {
		if constexpr (internal_limbs == 1) {
			__data[0] = neg_mod3_limb(__data[0]);
			return;
		}

		uint32_t i = 0;

		for (; i+2 <= internal_limbs; i += 2) {
			__uint128_t t = neg_mod3_limb128(*((__uint128_t *)&__data[i]));
			*((__uint128_t *)&__data[i]) = t;
		}

		for (; i < internal_limbs; i++) {
			__data[i] = neg_mod3_limb(__data[i]);
		}
	}

	/// \tparam k_lower lower limit inclusive
	/// \tparam k_upper uuper limit exclusive
	template<uint32_t k_lower, uint32_t k_upper>
	constexpr inline void neg() noexcept {
		static_assert(k_upper <= LENGTH && k_lower < k_upper);

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
	constexpr static inline T neg_mod3_limb(const T a) noexcept {
		// int(0b0101010101010101010101010101010101010101010101010101010101010101)
		constexpr T c1 = T(6148914691236517205u);
		//int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T c2 = T(12297829382473034410u);

		const T e1 = a&c1; // filter the ones
		const T e2 = a&c2; // filter the twos

		// re-shifts everything to the correct place
		return (e1 << 1) ^ (e2 >> 1);
	}

	/// negate a on every coordinate
	/// \param a input
	/// \return -a
	constexpr static inline __uint128_t neg_mod3_limb128(const __uint128_t a) noexcept {
		constexpr __uint128_t c2 = __uint128_t(12297829382473034410u)  << 64 | 12297829382473034410u;
		constexpr __uint128_t c1 = __uint128_t(6148914691236517205u)   << 64 | 6148914691236517205u;

		const __uint128_t e1 = a&c1;
		const __uint128_t e2 = a&c2;
		return (e1 << 1) ^ (e2 >> 1);
	}

#ifdef USE_AVX2
	/// negate a on every coordinate
	/// \param a input
	/// \return -a
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
	constexpr static inline T mod3_limb(const T a) noexcept {
		const T e = mod3_limb_withoutcorrection(a);

		// fix the overflow on coordinate 32
		constexpr T ofmask = (T(1u) << 62) - 1;
		const T ofbit = ((a>>62)%3) << 62;
		return (e&ofmask)^ofbit;
	}

	/// same as mod3_limb but without the correction of the last entry
	/// \param a
	/// \return
	constexpr static inline T mod3_limb_withoutcorrection(const T a) noexcept {
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

	/// NOTE: every coordinate can be at most 4
	/// \param a
	/// \return a mod 3, in every coordinate
	constexpr static inline __uint128_t mod3_limb128(const __uint128_t a) noexcept {
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
	constexpr static inline T sub_mod3_limb(const T x,
	                                        const T y) noexcept {
		return add_mod3_limb(x, neg_mod3_limb(y));
	}

	/// \param x: input
	/// \param y: input
	/// \return x-y, on every coordinate
	constexpr static inline __uint128_t sub_mod3_limb128(const __uint128_t x,
	                                                     const __uint128_t y) noexcept {
		return add_mod3_limb128(x, neg_mod3_limb128(y));
	}

#ifdef USE_AVX2
	/// \param x ; input
	/// \param y ; input
	/// \return x - y
	static inline __m256i sub_mod3_limb256(const __m256i x,
	                                       const __m256i y) noexcept {
		return add_mod3_limb256(x, neg_mod3_limb256(y));
	}
#endif

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
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);
		for (uint32_t i = k_lower; i < k_upper; i++) {
			int64_t data = int64_t(v1.get(i)) - int64_t(v2.get(i)) + MODULUS;
			v3.set(data % MODULUS, i);
		}
	}

	///
	/// \param v3 output: = v1 - v2
	/// \param v1 input:
	/// \param v2 input:
	constexpr inline static void sub(kAryPackedContainer_T &v3,
	                                 kAryPackedContainer_T const &v1,
	                                 kAryPackedContainer_T const &v2) noexcept {
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
		if constexpr(activate_avx2) {
			for (; i + 4 <= internal_limbs; i += 4) {
				__m256i t = sub_mod3_limb256(_mm256_lddqu_si256((__m256i *) &v1.__data[i]),
				                             _mm256_lddqu_si256((__m256i *) &v2.__data[i]));
				_mm256_storeu_si256((__m256i *) &v3.__data[i], t);
			}
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

	/// NOTE: this function, does not reduce anything.
	/// only call it if you know, that now overflow happens
	/// \tparam TT
	/// \param a input
	/// \param b input
	/// \return a+b
	template<typename TT>
	constexpr static inline TT add_mod3_limb_no_overflow(const TT a,
	                                                     const TT b) noexcept {
		// we know that no overflow will happen, So no 2+2=1
		return a+b;
	}

	/// \param x input first number
	/// \param y input second number
	/// \return x+y in every number
	constexpr static inline T add_mod3_limb(const T x,
	                                        const T y) noexcept {
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

	/// \param x input
	/// \param y input
	/// \return x+y mod 3, in every coordinate
	constexpr static inline __uint128_t add_mod3_limb128(const __uint128_t x,
	                                                     const __uint128_t y) noexcept {
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
	/// this function assumes that a,b \in [0,1], so there is no 2.
	/// \param a input
	/// \param b input
	/// \return a+b
	constexpr static inline __m256i add_mod3_limb_no_overflow(const __m256i a,
	                                                          const __m256i b) noexcept {
		// no overflow will happen.
		return _mm256_add_epi64(a,b);
	}

	/// \param x input
	/// \param y input
	/// \return x+y mod3
	static inline __m256i add_mod3_limb256(const __m256i x,
	                                       const __m256i y) noexcept {
		const static __m256i c1 = _mm256_set_epi64x(6148914691236517205ull,6148914691236517205ull,6148914691236517205ull,6148914691236517205ull);
		const static __m256i c2 = _mm256_set_epi64x(12297829382473034410ull,12297829382473034410ull,12297829382473034410ull,12297829382473034410ull);

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


#ifdef USE_AVX2
	/// \param v3 output = v1 + v2 mod3
	/// \param v1 input
	/// \param v2 input
	constexpr inline static void add(kAryPackedContainer_T &v3,
	                       			 kAryPackedContainer_T const &v1,
	                       			 kAryPackedContainer_T const &v2) noexcept {
		if constexpr (internal_limbs == 1) {
			v3.__data[0] = add_mod3_limb(v1.__data[0], v2.__data[0]);
			return;
		} else if constexpr (internal_limbs == 2) {
			__uint128_t t = add_mod3_limb128(*((__uint128_t *)v1.__data.data()), *((__uint128_t *)v2.__data.data()));
			*(__uint128_t *)v3.__data.data() = t;
			return;
		} else if constexpr (internal_limbs == 4) {
			__m256i t = add_mod3_limb256(_mm256_lddqu_si256((__m256i *)&v1.__data[0]), _mm256_lddqu_si256((__m256i *)&v2.__data[0]));
			_mm256_storeu_si256((__m256i *)&v3.__data[0], t);
			return;
		}

		uint32_t i = 0;
		if constexpr(activate_avx2) {
			for (; i + 4 <= internal_limbs; i += 4) {
				__m256i t = add_mod3_limb256(_mm256_lddqu_si256((__m256i *) &v1.__data[i]),
				                             _mm256_lddqu_si256((__m256i *) &v2.__data[i]));
				_mm256_storeu_si256((__m256i *) &v3.__data[i], t);
			}
		}

		for (; i+2 <= internal_limbs; i += 2) {
			__uint128_t t = add_mod3_limb128(*((__uint128_t *)&v1.__data[i]), *((__uint128_t *)&v2.__data[i]));
			*((__uint128_t *)&v3.__data[i]) = t;
		}

		for (; i < internal_limbs; i++) {
			v3.__data[i] = add_mod3_limb(v1.__data[i], v2.__data[i]);
		}
	}
#endif
	/// optimised version of the function above.
	/// \tparam l lower limit, inclusive
	/// \tparam h upper limit, exclusive
	/// \param v3 output = v1 + v2, between [l, h)
	/// \param v1 input
	/// \param v2 input
	/// \return hamming weight
	template<const uint32_t l, const uint32_t h>
	constexpr static uint16_t add_only_weight_partly(kAryPackedContainer_T &v3,
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

		// add everything that is left
		for (uint32_t i = hlimb+1; i < internal_limbs; i++) {
			v3.__data[i] = add_mod3_limb(v1.__data[i], v2.__data[i]);
		}

		return weight;
	}

	/// \param a input
	/// \return 2*a, input must be reduced mod 3
	constexpr static inline T times2_mod3_limb(const T a) noexcept {
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
	constexpr static inline bool filter2_mod3_limb(const T a) noexcept {
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		return (a&m) != 0;
	}

	/// \param a element to check
	/// \param limit
	/// \return returns the number of two in the limb
	constexpr static inline uint32_t filter2count_mod3_limb(const T a) noexcept {
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410ull);
		return __builtin_popcountll(a&m);
	}

	///
	/// \param a
	/// \return
	constexpr static inline uint32_t filter2count_mod3_limb128(const __uint128_t a) noexcept {
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
	constexpr static inline uint32_t filter2count_range_mod3_limb(const T a) noexcept {
		static_assert(k_lower != 0 && k_lower < k_upper && k_upper <= LENGTH);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		constexpr T mask = ((T(1u) << (2u*k_lower)) - 1u) & ((T(1u) << (2u*k_upper)) - 1u);
		return __builtin_popcountll(a&mask&m);
	}

	/// counts the number of twos upto `k_upper` (exclusive)
	/// \tparam kupper
	template<const uint16_t k_upper>
	constexpr inline uint32_t filter2count_range_mod3() {
		static_assert(k_upper <= LENGTH);
		// int(0b1010101010101010101010101010101010101010101010101010101010101010)
		constexpr T m = T(12297829382473034410u);
		constexpr T mask = (T(1u) << (2u*k_upper)%bits_per_limb) - 1u;
		constexpr uint32_t limb = std::max(1, (k_upper+numbers_per_limb-1)/ numbers_per_limb);

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

	/// counts the number of twos in a single limb
	/// upto `k_upper` (exclusive)
	/// \tparam k_upper upper limit
	/// \param a input
	/// \return number of twos
	template<const uint16_t k_upper>
	constexpr static inline uint32_t filter2count_range_mod3_limb(const T a) noexcept {
		ASSERT(0 < k_upper && k_upper <= LENGTH);
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
		for (; i < internal_limbs; i++){
			v3.__data[i] = times2_mod3_limb(v1.__data[i]);
		}
	}


	/// TODO make this available for add/sub/mul/scalar/ and the same for avx
	/// \tparam T
	/// \param in1
	/// \param in2
	/// \return
	template<typename TT>
	constexpr inline static TT add_T(const TT in1, const TT in2) {
		// TODO
	}




	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__  static constexpr bool optimized() noexcept { return true; };
};


///
/// \tparam T
/// \tparam n
/// \tparam q
/// \param out
/// \param obj
/// \return
template<typename T, const uint32_t n, const uint32_t q>
std::ostream& operator<< (std::ostream &out, const kAryPackedContainer_T<T, n, q> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << unsigned (obj[i]);
	}
	return out;
}
#endif
