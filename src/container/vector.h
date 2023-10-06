#ifndef CRYPTANALYZELIB_CONTAINER_VECTOR
#define CRYPTANALYZELIB_CONTAINER_VECTOR

#include <cstdint>
#include "helper.h"
#include "container/common.h"


/// simple data container holding `length` Ts
/// \tparam T base type
/// \tparam length number of elements
template<class T, uint32_t length>
	requires kAryContainerAble<T>
class Vector {
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
	inline static void add(Vector &v3,
	                       Vector const &v1,
	                       Vector const &v2,
	                       const uint32_t k_lower=0,
	                       const uint32_t k_upper=LENGTH) noexcept {
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
	inline static bool add(Vector &v3,
	                       Vector const &v1,
	                       Vector const &v2,
	                       const uint32_t k_lower,
	                       const uint32_t k_upper,
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
	inline static void sub(Vector &v3,
	                       Vector const &v1,
	                       Vector const &v2,
	                       const uint32_t k_lower=0,
	                       const uint32_t k_upper=LENGTH) noexcept {
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
	inline static bool sub(Vector &v3,
	                       Vector const &v1,
	                       Vector const &v2,
	                       const uint32_t k_lower,
	                       const uint32_t k_upper,
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
	inline static bool cmp(Vector const &v1,
	                       Vector const &v2,
	                       const uint32_t k_lower=0,
	                       const uint32_t k_upper=LENGTH) noexcept {
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
	inline static void set(Vector &v1,
	                       Vector const &v2,
	                       const uint32_t k_lower=0,
	                       const uint32_t k_upper=LENGTH) noexcept {
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
	bool is_equal(Vector const &obj,
	              const uint32_t k_lower=0,
	              const uint32_t k_upper=LENGTH) const noexcept {
		return cmp(this, obj, k_lower, k_upper);
	}

	/// \param obj to compare to
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	/// \return this > obj on the coordinates [k_lower, k_higher)
	bool is_greater(Vector const &obj,
	                const uint32_t k_lower=0,
	                const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= length);
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
	/// \param k_lower lower bound coordinate wise
	/// \param k_upper higher bound coordinate wise
	/// \return this < obj on the coordinates [k_lower, k_higher)
	bool is_lower(Vector const &obj,
	              const uint32_t k_lower=0,
	              const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_upper <= length);
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

	/// access operator
	/// \param i position. Boundary check is done.
	/// \return limb at position i
	T& operator [](size_t i) noexcept {
		ASSERT(i < length);
		return __data[i];
	}
	const T& operator [](const size_t i) const noexcept {
		ASSERT(i < length);
		return __data[i];
	};

	/// copy operator
	/// \param obj to copy from
	/// \return this
	Vector& operator =(Vector const &obj) noexcept {
		ASSERT(size() == obj.size() && "äh?");

		if (likely(this != &obj)) { // self-assignment check expected
			__data = obj.__data;
		}

		return *this;
	}

	/// prints this container between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound
	/// \param k_upper higher bound
	void print(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < length && k_upper < length && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << __data[i] << " ";
		}
		std::cout << "\n";
	}

	/// prints this container between the limbs [k_lower, k_higher)
	/// \param k_lower lower bound
	/// \param k_upper higher bound
	void print_binary(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		ASSERT(k_lower < length && k_upper < length && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			auto data = __data[i];
			for (uint32_t j = 0; j < T::bits(); ++j) {
				std::cout << (data & 1u) << " ";
				data >>= 1u;
			}
		}
		std::cout << "\n";
	}
	
	/// iterators
	auto begin() noexcept { return __data.begin();}
	auto begin() const noexcept { return __data.begin();}
	auto end() noexcept { return __data.end();}
	auto end() const noexcept { return __data.end();}

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
#endif
