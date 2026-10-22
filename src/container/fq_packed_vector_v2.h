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
	constexpr static uint32_t limb_bits = sizeof(T) * 8u;
	constexpr static T number_mask = (bits_per_number == limb_bits) ? T(-1ull) : T((T(1u) << bits_per_number) - 1u);

	// Number of Limbs needed to represent `length` numbers of `bits_per_number`
	// bits. NOTE: the numbers are packed densely, so a number can be split
	// over two limbs.
	constexpr static uint16_t internal_limbs = (n * bits_per_number + limb_bits - 1u) / limb_bits;
    // true if we need every bit of the last limb
	constexpr static bool is_full = ((n * bits_per_number) % limb_bits) == 0;

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
	constexpr FqPackedVectorMeta_v2(const DataType a) noexcept : __data() {
		set(a, 0);
	}

	/// \param b[in]: first bit
	/// \param len[in]: number of bits, <= 64
	/// \return the bits [b, b+len) of the vector, shifted down
	[[nodiscard]] constexpr inline uint64_t get_bits(const uint32_t b,
	                                                 const uint32_t len) const noexcept {
		assert(len <= 64u);
		assert(b + len <= internal_limbs * limb_bits);
		uint64_t ret = 0;
		uint32_t done = 0;
		while (done < len) {
			const uint32_t p = b + done;
			const uint32_t off = p / limb_bits, sh = p % limb_bits;
			const uint32_t take = ((limb_bits - sh) < (len - done)) ? (limb_bits - sh) : (len - done);
			const uint64_t m = (take >= 64u) ? uint64_t(-1ull) : ((1ull << take) - 1ull);
			ret |= ((uint64_t(__data[off]) >> sh) & m) << done;
			done += take;
		}
		return ret;
	}

    /// \tparam l[in]: lower bound (inclusive)
    /// \tparam h[in]: upper bound (exclusive)
	/// \return the numbers [l, h), as a bit string
	template<const uint32_t l,
             const uint32_t h>
	[[nodiscard]] constexpr inline uint64_t hash() const noexcept {
		static_assert(l < h);
		static_assert(h <= n);
		static_assert(is_hashable<l, h>());
		return get_bits(l * bits_per_number, (h - l) * bits_per_number);
    }

	/// \param l[in]: lower bound (inclusive)
	/// \param h[in]: upper bound (exclusive)
	/// \return the numbers [l, h), as a bit string
	[[nodiscard]] constexpr inline uint64_t hash(const uint32_t l,
	                                             const uint32_t h) const noexcept {
		assert(l < h);
		assert(h <= n);
		return get_bits(l * bits_per_number, (h - l) * bits_per_number);
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
	/// \return the number you wanted to access, shifted down to the lowest
	/// 	bits. If not `__unsigned`: in [-q/2, q/2]
	[[nodiscard]] constexpr inline DataType get(const uint32_t i) const noexcept {
        assert(i < length);
		const uint32_t p = i * bits_per_number;
		const uint32_t off = p / limb_bits, sh = p % limb_bits;
		T t = __data[off] >> sh;
		// the number is split over two limbs
		if ((sh + bits_per_number) > limb_bits) {
			t |= __data[off + 1u] << (limb_bits - sh);
		}
		t &= number_mask;

		if constexpr (!__unsigned) {
			if (t > (q / 2u)) {
				return DataType(int64_t(t) - int64_t(q));
			}
		}
		return DataType(t);
    }

	/// sets the i-th coordinate to `data` mod q
	/// \param data[in]: number to set
	/// \param i[in]: coordinate
    constexpr inline void set(const DataType data,
	                          const uint32_t i) noexcept {
		assert(i < length);
		T v;
		if constexpr (!__unsigned) {
			int64_t d = int64_t(data) % int64_t(q);
			if (d < 0) { d += int64_t(q); }
			v = T(d);
		} else {
			v = T(uint64_t(data) % q);
		}

		const uint32_t p = i * bits_per_number;
		const uint32_t off = p / limb_bits, sh = p % limb_bits;
		__data[off] = (__data[off] & ~T(number_mask << sh)) | T(v << sh);
		// the number is split over two limbs
		if ((sh + bits_per_number) > limb_bits) {
			const uint32_t s = limb_bits - sh;
			const T m = number_mask >> s;
			__data[off + 1u] = (__data[off + 1u] & ~m) | T(v >> s);
		}
    }

private:
	std::array<T, internal_limbs> __data;
};
