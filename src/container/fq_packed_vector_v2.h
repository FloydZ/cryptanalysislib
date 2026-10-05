#pragma once 

#include "container/fq_packed_vector.h"


/// represents a vector of numbers mod `_q` in vector of `_T` in a compressed way
/// Meta class, contains all important meta definitions.
/// \tparam _n = number of elements
/// \tparam _q = modulus
/// \tparam __unsigned:
///         if true the elements will be represented as numbers within [0,...,q)
///         if false the elements will be represented as numbers within 
///          [-q/2,...(q/1)-1], except if q==2. But in this case due to partial 
///         specialization the optimized class `BinaryContainer` is used.
///         NOTE: this config flag is not part of the `config`, as partial 
///         specialized classes need to access it.
///         NOTE: internally the numbers are still computed and stored within
///         [0,...,q), only if you access it, they will be translated into
///         their signed form.
/// \tparam config: config class
template<const uint32_t _n,
		 const uint64_t _q,
         typename T=uint64_t,
         const bool __unsigned =  true,
		 const FqPackedVectorMetaConfig &config=fqPackedVectorMetaConfig>
#if __cplusplus > 201709L
    requires std::is_integral_v<T> &&
    		 std::is_unsigned_v<T>
#endif
class FqPackedVectorMeta_v2 {
public:
    
	// make the length and modulus of the container public available
	constexpr static uint64_t q = _q;
	constexpr static uint64_t modulus = q;
	constexpr static uint32_t n = _n;
	constexpr static uint64_t length = n;
	
	static_assert(n > 0, "jeah at least a single bit?");
	static_assert(q > 1, "mod 1 or 0?");
	static_assert(ceil_log2(q) <= (8*sizeof(T)), 
                  "the limb type should be atleast of the size of prime");
	
    constexpr static uint32_t bits_per_number = (uint32_t) ceil_log2(q);
	
	// Number of Limbs needed to represent `length` numbers of size log(MOD) +1
	constexpr static uint16_t internal_limbs = (n + bits_per_number - 1) / bits_per_number;
    // true if we need every bit of the last limb
	constexpr static bool is_full = (n%bits_per_number) == 0;

	//
	constexpr static bool activate_simd = config.activate_simd;
	using S = SIMDSelector<T>;
	constexpr static uint16_t numbers_per_simd_limb = (sizeof(S) * 8) / bits_per_number;

	constexpr static uint32_t total_bytes = sizeof(T) * internal_limbs;
	constexpr static uint32_t total_bits =  total_bytes * 8;

	// we are good C++ devs.
	typedef T ContainerLimbType;
	using DataType = LogTypeTemplate<bits_per_number, __unsigned>;

	// list compatibility typedef
	typedef T LimbType;
	typedef T LabelContainerType;


	// this will zero initialize everything, i think
	constexpr FqPackedVectorMeta_v2() noexcept : __data() {}
	constexpr FqPackedVectorMeta_v2(const FqPackedVectorMeta_v2 &a) noexcept = default;

	/// just a wrapper, for testing single elements and not arrays
	constexpr FqPackedVectorMeta_v2(const DataType a) noexcept {
		DataType t = a;
		if constexpr (!__unsigned) {
			while (t < 0) { t += q; }
		}
		t = t%q;
		__data[0] = t;
	}

    /// \tparam l[in]: lower bound (inclusive)
    /// \tparam h[in]: upper bound (exclusive)
	template<const uint32_t l, 
             const uint32_t h>
	[[nodiscard]] constexpr inline auto hash() const noexcept {
        // TODO 
        return 0;
    }

	[[nodiscard]] constexpr inline auto hash(const uint32_t l,
	                                         const uint32_t h) const noexcept {
        return 0; // TODO
    }

	// simple hash function
	[[nodiscard]] constexpr inline auto hash() const noexcept {
		return *this;
		// return Hash<uint64_t, 0, n, q>::hash((uint64_t *)ptr());
	}

	/// \tparam l
	/// \tparam h
	/// \return true if h-l <= 64, which is the maximum
	///		this implementation can use as a hash for
	///		a hashmap or sorting/searching
	template<const uint32_t l, const uint32_t h>
	constexpr static bool is_hashable() noexcept {
		static_assert(h > l);
		constexpr size_t t1 = h-l;
		constexpr size_t t2 = t1*bits_per_number;

		return t2 <= 64u;
    }

	/// access the i-th coordinate/number/elemen
	/// \param i coordinate to access.
	/// \return the number you wanted to access, shifted down to the lowest bits.
	[[nodiscard]] constexpr inline DataType get(const uint32_t i) const noexcept {
        assert(i < length);
        const uint32_t a1 = (i +              0) * internal_limbs;
        const uint32_t a2 = (i + internal_limbs) * internal_limbs;        
        const uint32_t off1 = a1 / (sizeof(T) * 8u);
        const uint32_t off2 = a2 / (sizeof(T) * 8u);
    
        const uint32_t shift1 = a1 % (sizeof(T) * 8);
        const uint32_t shift2 = (sizeof(T) * 8) - (a1 % (sizeof(T) * 8));

        const T m1 = (off1 != off2) * T(-1ull);
        const T t1 = __data[off1] >> shift1;
        const T t2 = (__data[off2] << shift2) & m1;

        const T ret = t1 | t2;
        return ret;
    }
	
    constexpr inline void set(const DataType data,
	                          const uint32_t i) noexcept {
    }

private:
	std::array<T, internal_limbs> __data;
};
