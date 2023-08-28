#ifndef SMALLSECRETLWE_FQ_VECTOR_H
#define SMALLSECRETLWE_FQ_VECTOR_H

#include <array>
#include <cstdint>
#include <atomic>

#include "helper.h"
#include "random.h"
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

/// simple data container holding `length` Ts
/// \tparam T base type
/// \tparam length number of elements
template<class T, const uint32_t n, const uint32_t q>
	requires kAryContainerAble<T>
class kAryContainer_T {
public:

	// Needed for the internal template system.
	typedef T DataType;
	typedef T ContainerLimbType;

	// internal data length. Used in the template system to pass through this information
	constexpr static uint32_t LENGTH = n;
	constexpr static uint32_t MODULUS = q;

	/// zeros our the whole container
	/// \return nothing
	constexpr inline void zero() noexcept {
		LOOP_UNROLL();
		for (unsigned int i = 0; i < LENGTH; i++){
			__data[i] = T(0);
		}
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
	[[nodiscard]] bool is_zero() const noexcept {
		for (uint32_t i = 0; i < LENGTH; ++i) {
			if(__data[i] != T(0))
				return false;
		}

		return true;
	}

	/// calculate the hamming weight
	/// \return the hamming weight
	[[nodiscard]] inline uint32_t weight() const noexcept {
		uint32_t r = 0;
		for (uint32_t i = 0; i < LENGTH; ++i) {
			if (__data[i] != 0)
				r += 1;
		}
		return r;
	}

	/// swap coordinate i, j, boundary checks are done
	/// \param i coordinate
	/// \param j coordinate
	void swap(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < LENGTH && j < LENGTH);
		SWAP(__data[i], __data[j]);
	}

	/// *-1
	/// \param i
	void flip(const uint32_t i) noexcept {
		ASSERT(i < LENGTH);
		__data[i] *= -1;
	}

	/// negate every coordinate between [k_lower, k_higher)
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	inline void neg(const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= LENGTH &&k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			__data[i] = ((T(0) - __data[i]) + q) % q;
		}
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	inline static void mul(kAryContainer_T &v3,
					const kAryContainer_T &v1,
					const kAryContainer_T &v2,
	                const uint32_t k_lower=0,
					const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

		LOOP_UNROLL();
		for (uint32_t i = k_lower; i < k_upper; ++i) {
			v3.__data[i] = (v1.__data[i] * v2.__data[i]) % q;
		}
	}

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	inline static void add(kAryContainer_T &v3,
					const kAryContainer_T &v1,
					const kAryContainer_T &v2,
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
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \param norm = max norm of an dimension which is allowed.
	/// \return true if the element needs to be filtered out. False else.
	inline static bool add(kAryContainer_T &v3,
					kAryContainer_T const &v1,
					kAryContainer_T const &v2,
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

	/// \param v3 output container
	/// \param v1 input container
	/// \param v2 input container
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \return true if the elements needs to be filled out. False else.
	inline static void sub(kAryContainer_T &v3,
					kAryContainer_T const &v1,
					kAryContainer_T const &v2,
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
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \param norm filter every element out if hte norm is bigger than `norm`
	/// \return true if the elements needs to be filter out. False if not
	inline static bool sub(kAryContainer_T &v3, 
					kAryContainer_T const &v1, 
					kAryContainer_T const &v2,
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
	/// \param k_lower lower dimension
	/// \param k_upper higher dimension
	/// \return v1 == v2 on the coordinates [k_lower, k_higher)
	inline static bool cmp(kAryContainer_T const &v1, 
					kAryContainer_T const &v2,
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
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	inline static void set(kAryContainer_T &v1,
					kAryContainer_T const &v2,
	                const uint32_t k_lower=0, 
					const uint32_t k_upper=LENGTH) noexcept {
		ASSERT(k_upper <= LENGTH && k_lower < k_upper);

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
	              const uint32_t k_lower=0,
				  const uint32_t k_upper=LENGTH) const noexcept {
		return cmp(this, obj, k_lower, k_upper);
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	/// \return this > obj on the coordinates [k_lower, k_higher)
	bool is_greater(kAryContainer_T const &obj,
	                const uint32_t k_lower=0,
					const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= LENGTH && "ERROR is_greater not correct k_upper");
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
	              const uint32_t k_lower=0,
				  const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= LENGTH && "ERROR is_lower not correct k_upper");
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
	T& operator [](const size_t i) noexcept {
		ASSERT(i < LENGTH && "wrong access index");
		return __data[i];
	}
	const T& operator [](const size_t i) const noexcept {
		ASSERT(i < LENGTH && "wrong access index");
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

		// TODO correct?

		return *this;
	}

	/// prints this container between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound
	/// \param k_upper higher bound
	void print(const uint32_t k_lower, const uint32_t k_upper) const noexcept {
		ASSERT(k_lower < LENGTH && k_upper < LENGTH && k_lower < k_upper);
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
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return LENGTH; }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return LENGTH; }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return LENGTH*sizeof(T); }

	/// returns the underlying data container
	__FORCEINLINE__ std::array<T, LENGTH>& data() noexcept { return __data; }
	__FORCEINLINE__ const std::array<T, LENGTH>& data() const noexcept { return __data; }
	//T& data(const uint64_t index) { ASSERT(index < length && "wrong index"); return __data[index]; }
	const T data(const uint64_t index) const noexcept { ASSERT(index < LENGTH && "wrong index"); return __data[index]; }
	const T get(const uint64_t index) const noexcept { ASSERT(index < LENGTH && "wrong index"); return __data[index]; }

	/// TODO remove
	T get_type() noexcept {return __data[0]; }
protected:
	std::array<T, LENGTH> __data;
};

/// specialized class for representing stuff on 8 bit
/// NOTE: this implements the representation padded (e.g. every number gets 8 bit)
template<const uint32_t n>
	requires kAryContainerAble<uint8_t>
class kAryContainer_T<uint8_t, n, 4> : public kAryContainer_T<uint64_t, n, 4> {
	/// this class implements the following representation
	///
	/// [ ||| ]
	///
	///
	///
	///

	// external ressources
	using kAryContainer_T<uint64_t, n, 4>::__data;

	// helper sizes
	static constexpr uint32_t limb_u32 	= (n + 	3u)/ 4u;
	static constexpr uint32_t limb_u64 	= (n + 	7u)/ 8u;
	static constexpr uint32_t limb_u128 = (n + 15u)/16u;
	static constexpr uint32_t limb_u256 = (n + 31u)/32u;
	
	// helper pointers
	uint32_t *__data32 = (uint32_t *)__data.data();
	uint64_t *__data64 = (uint64_t *)__data.data();
	__uint128_t *__data128 = (__uint128_t *)__data.data();
#ifdef USE_AVX2
	__m256i *__data256 = (__m256i *)__data.data();
#endif

	// helper masks
	static constexpr __uint128_t mask_4 = (__uint128_t(0x0303030303030303ULL) << 64UL) | (__uint128_t(0x0303030303030303ULL));
#ifdef USE_AVX2
	static constexpr __m256i mask256_4 = _mm256_set();
#endif

	public:

	/// 
	template<typename T>
	static inline T mod_T(const T a) noexcept {
		return a & mask_4;
	}

	///
	template<typename T>
	static inline T add_T(const T a, const T b) noexcept {
		return (a + b) & mask_4;
	}

	///
	template<typename T>
	static inline T mul_T(const T a, const T b) noexcept {
		/// TODO
		return 0;
	}

#ifdef USE_AVX2
#endif
};

#endif
