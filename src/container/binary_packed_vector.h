#ifndef CRYPTANALYSISLIB_BINARYPACKEDCONTAINER_H
#define CRYPTANALYSISLIB_BINARYPACKEDCONTAINER_H

#include <array>
#include <atomic>
#include <cstdint>

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

// local includes
#include "helper.h"
#include "popcount/popcount.h"
#include "random.h"
#include "simd/simd.h"

// C macro for implementing multi limb comparison.
#define BINARYCONTAINER_COMPARE(limb1, limb2, op1, op2) \
	if (limb1 op1 limb2)                                \
		return 1;                                       \
	else if (limb1 op2 limb2)                           \
		return 0;

// C Macro for implementing multi limb comparison
#define BINARYCONTAINER_COMPARE_MASKED(limb1, limb2, mask, op1, op2) \
	if ((limb1 & mask) op1(limb2 & mask))                            \
		return 1;                                                    \
	else if ((limb1 & mask) op2(limb2 & mask))                       \
		return 0;


/// \tparam _n number of bits
/// \tparam T
template<const uint64_t _n, 
		 typename T = uint64_t>
#if __cplusplus > 201709L
    requires std::is_arithmetic_v<T>
#endif
class BinaryContainer {
public:
	// Internal Types needed for the template system.
	typedef BinaryContainer<_n, T> ContainerType;
	typedef T LimbType;
	typedef bool DataType;

	// internal data length. Need to export it for the template system.
	constexpr static uint64_t n = _n;
	[[nodiscard]] constexpr static inline uint64_t length() noexcept { return n; }
	constexpr static uint64_t q = 2;
	[[nodiscard]] constexpr static inline uint64_t modulus() noexcept { return q; }

	constexpr static uint32_t RADIX = sizeof(T) * 8;
	constexpr static T minus_one = T(-1);

	static_assert(_n > 0);
public:
	// how many limbs to we need and how wide are they.
	[[nodiscard]] constexpr static uint16_t limb_bits_width() noexcept { return limb_bytes_width() * 8; };
	[[nodiscard]] constexpr static uint16_t limb_bytes_width() noexcept { return sizeof(T); };

	//private:
	// DO NOT CALL THIS FUNCTION. Use 'limbs()'.
	constexpr static uint16_t compute_limbs() noexcept {
#ifdef BINARY_CONTAINER_ALIGNMENT
		return (alignment() + limb_bits_width() - 1) / limb_bits_width();
#else
		return (length() + limb_bits_width() - 1) / limb_bits_width();
#endif
	};

public:
	/// default constructor
	constexpr BinaryContainer() noexcept : __data() {}

	/// Copy Constructor
	constexpr BinaryContainer(const BinaryContainer &a) noexcept : __data(a.__data) {}


	// round a given amount of 'in' bits to the nearest limb excluding the lowest overflowing bits
	// e.g. 13 -> 64
	[[nodiscard]] constexpr static uint16_t round_up(uint16_t in) noexcept { return round_up_to_limb(in) * limb_bits_width(); }
	[[nodiscard]] constexpr static uint16_t round_up_to_limb(uint16_t in) noexcept { return (in / limb_bits_width()) + 1; }

	// the same as above only rounding down
	// 13 -> 0
	[[nodiscard]] constexpr static uint16_t round_down(uint16_t in) noexcept { return round_down_to_limb(in) * limb_bits_width(); }
	[[nodiscard]] constexpr static uint16_t round_down_to_limb(uint16_t in) { return (in / limb_bits_width()); }

	// calculate from a bit-position 'i' the mask to set it.
	[[nodiscard]] constexpr static T mask(uint16_t i) noexcept {
		ASSERT(i <= length() && "wrong access index");
		T u = i % limb_bits_width();
		return (T(1) << u);
	}

	// same as the function below, but catches the special case when i == 0 %64.
	[[nodiscard]] constexpr static T lower_mask2(const uint16_t i) noexcept {
		ASSERT(i <= length());
		T u = i % limb_bits_width();
		if (u == 0) return T(-1);
		return ((T(1) << u) - 1);
	}

	// given the i-th bit this function will return a bits mask where the lower 'i' bits are set. Everything will be
	// realigned to limb_bits_width().
	[[nodiscard]] constexpr static T lower_mask(const uint16_t i) noexcept {
		ASSERT(i <= length());
		return ((T(1) << (i % limb_bits_width())) - 1);
	}

	// given the i-th bit this function will return a bits mask where the higher (n-i)bits are set.
	[[nodiscard]] constexpr static T higher_mask(const uint16_t i) noexcept {
		ASSERT(i <= length());
		// TODO better formula
		if ((i % limb_bits_width()) == 0) {
			return T(-1);
		}

		return (~((T(1u) << (i % limb_bits_width())) - 1));
	}

	// given the i-th bit this function will return a bits mask where the lower 'n-i' bits are set. Everything will be
	// realigned to limb_bits_width().
	[[nodiscard]] constexpr static T lower_mask_inverse(const uint16_t i) noexcept {
		ASSERT(i <= length() && "wrong access index");
		T u = i % limb_bits_width();

		if (u == 0) {
			return -1;
		}

		auto b = (T(1) << (limb_bits_width() - u));
		return b - T(1);
	}

	// given the i-th bit this function will return a bits mask where the higher (i) bits are set.
	[[nodiscard]] constexpr static inline  T higher_mask_inverse(const uint16_t i) noexcept {
		ASSERT(i <= length() && "wrong access index");
		return ~lower_mask_inverse(i);
	}

	// not shifted.
	[[nodiscard]] constexpr inline T get_bit(const uint16_t i) const noexcept {
		return __data[round_down_to_limb(i)] & mask(i);
	}

	// shifted.
	[[nodiscard]] constexpr inline bool get_bit_shifted(const uint16_t i) const noexcept {
		return (__data[round_down_to_limb(i)] & mask(i)) >> (i % RADIX);
	}

	// return the bits [i,..., j) in one limb
	[[nodiscard]] constexpr inline T get_bits(const uint16_t i,
	                            const uint16_t j) const noexcept {
		ASSERT(j > i && j - i <= limb_bits_width() && j <= length());
		const T lmask = higher_mask(i);
		const T rmask = lower_mask2(j);
		const int64_t lower_limb = i / limb_bits_width();
		const int64_t higher_limb = (j - 1) / limb_bits_width();

		const uint64_t shift = i % limb_bits_width();
		if (lower_limb == higher_limb) {
			return (__data[lower_limb] & lmask & rmask) >> (shift);
		} else {
			const T a = __data[lower_limb] & lmask;
			const T b = __data[higher_limb] & rmask;

			auto c = (a >> shift);
			auto d = (b << (limb_bits_width() - shift));
			auto r = c ^ d;
			return r;
		}
	}

	/// call like this
	/// const T lmask = higher_mask(i);
	/// const T rmask = lower_mask2(j);
	/// const int64_t lower_limb = i / limb_bits_width();
	/// const int64_t higher_limb = (j - 1) / limb_bits_width();
	/// const uint64_t shift = i % limb_bits_width();
	[[nodiscard]] constexpr inline T get_bits(const uint64_t llimb,
	                            const uint64_t rlimb,
	                            const uint64_t lmask,
	                            const uint64_t rmask,
	                            const uint64_t shift) const noexcept {
		if (llimb == rlimb) {
			return (__data[llimb] & lmask & rmask) >> (shift);
		} else {
			const T a = __data[llimb] & lmask;
			const T b = __data[rlimb] & rmask;

			auto c = (a >> shift);
			auto d = (b << (limb_bits_width() - shift));
			auto r = c ^ d;
			return r;
		}
	}

	constexpr inline void write_bit(const uint32_t pos,
	                                const bool b) noexcept {
		__data[pos / RADIX] = ((__data[pos / RADIX] & ~(1ull << (pos % RADIX))) | (T(b) << (pos % RADIX)));
	}

	inline constexpr void set_bit(const uint32_t pos) noexcept {
		__data[round_down_to_limb(pos)] |= (T(1) << (pos));
	}

	inline constexpr void flip_bit(const uint32_t pos) noexcept {
		__data[round_down_to_limb(pos)] ^= (T(1) << (pos));
	}

	inline constexpr void clear_bit(const uint32_t pos) noexcept {
		__data[round_down_to_limb(pos)] &= ~(T(1) << (pos));
	}

	/// zero the complete data vector
	constexpr void zero() noexcept {
		LOOP_UNROLL();
		for (uint32_t i = 0; i < limbs(); ++i) {
			__data[i] = 0;
		}
	}

	/// zeros the vector between [k_lower, k_upper)
	/// \param k_lower lower limit, inclusive
	/// \param k_upper upper limit, exclusive
	constexpr void zero(const uint32_t k_lower,
	                    const uint32_t k_upper) noexcept {
		const uint64_t lower = round_down_to_limb(k_lower);
		const uint64_t upper = round_down_to_limb(k_upper);

		const T lmask = higher_mask(k_lower);
		const T umask = k_upper % 64 == 0 ? T(0) : lower_mask(k_upper);

		if (lower == upper) {
			const T mask = ~(lmask & umask);
			__data[lower] &= mask;
			return;
		}

		__data[lower] &= ~lmask;
		__data[upper] &= ~umask;

		LOOP_UNROLL();
		for (uint32_t i = lower + 1; i < upper; ++i) {
			__data[i] = 0;
		}
	}

	/// seth the whole const_array to 'fff...fff'
	void one() noexcept {
		for (uint32_t i = 0; i < limbs(); ++i) {
			__data[i] = ~(__data[i] & 0);
		}
	}

	/// sets the vector to 0xff between [k_lower, k_upper)
	/// \param k_lower lower limit inclusive
	/// \param k_upper upper limit exclusive
	void one(const uint32_t k_lower,
	         const uint32_t k_upper) noexcept {
		const uint64_t lower = round_down_to_limb(k_lower);
		const uint64_t upper = round_down_to_limb(k_upper);

		const T lmask = higher_mask(k_lower);
		const T umask = k_upper % 64 == 0 ? T(0) : lower_mask(k_upper);

		if (lower == upper) {
			const T mask = (lmask & umask);
			__data[lower] |= minus_one & mask;
			return;
		}

		__data[lower] |= minus_one & lmask;
		__data[upper] |= minus_one & umask;

		for (uint32_t i = lower + 1; i < upper; ++i) {
			__data[i] = ~(__data[i] & 0);
		}
	}

	/// split the full length BinaryContainer into `k` windows.
	/// Inject in every window weight `w` on random positions.
	void random_with_weight_per_windows(const uint64_t w,
	                                    const uint64_t k) noexcept {
		std::vector<uint64_t> buckets_windows{};

		// this stupid approach needs to be done, because if w is not dividing n the last bits would be unused.
		buckets_windows.resize(k + 1);
		for (uint64_t i = 0; i < k; ++i) {
			buckets_windows[i] = i * length() / k;
		}
		buckets_windows[k] = length();

		// clear everything.
		zero();

		// for every window.
		for (uint64_t i = 0; i < k; ++i) {
			uint64_t cur_offset = buckets_windows[i];
			uint64_t windows_length = buckets_windows[i + 1] - buckets_windows[i];

			for (uint64_t j = 0; j < w; ++j) {
				write_bit(cur_offset + j, true);
			}

			// now permute
			for (uint64_t l = 0; l < windows_length; ++l) {
				uint64_t pos = fastrandombytes_uint64() % (windows_length - l);
				auto t = get_bit_shifted(cur_offset + l);
				write_bit(cur_offset + l, get_bit_shifted(cur_offset + l + pos));
				write_bit(cur_offset + l + pos, t);
			}
		}
	}

	///
	/// \param w weight to enumerated
	/// \param w max length <= length() over which the weight should be enumerated
	void random_with_weight(const uint32_t w,
	                        const uint32_t m=length(),
	                        const uint32_t offset=0) noexcept {
		ASSERT(m+offset <= length());
		ASSERT(w <= m);
		zero();

		for (uint64_t i = 0; i < w; ++i) {
			write_bit(i, true);
		}

		// early exit
		if (w == m) { return; }

		// now permute
		for (uint64_t i = 0; i < m; ++i) {
			uint64_t pos = fastrandombytes_uint64() % (m - i);
			bool t = get_bit_shifted(i+offset);
			write_bit(i+offset, get_bit_shifted(i + pos + offset));
			write_bit(i + pos + offset, t);
		}
	}

	/// set the whole data const_array on random data.
	void random() noexcept {
		constexpr uint64_t apply_mask = length() % limb_bits_width() == 0 ? lower_mask(length()) - 1 : lower_mask(length());

		if constexpr (length() < 64) {
			__data[0] = fastrandombytes_uint64() & apply_mask;
		} else {
			for (uint32_t i = 0; i < limbs() - 1; ++i) {
				__data[i] = fastrandombytes_uint64();
			}
			__data[limbs() - 1] = fastrandombytes_uint64() & apply_mask;
		}
	}

	[[nodiscard]] constexpr inline bool is_zero() const noexcept {
		for (uint32_t i = 0; i < limbs(); ++i) {
			if (__data[i] != 0) {
				return false;
			}
		}

		return true;
	}

	/// checks whether the vector is zero between [k_lower, k_upper)
	/// \param lower lower limit, inclusive
	/// \param upper upper limit, exclusive
	/// \return
	[[nodiscard]] constexpr inline bool is_zero(const uint32_t lower,
	                                            const uint32_t upper) const noexcept {
		ASSERT(upper <= length());
		const size_t lower_limb = round_down_to_limb(lower);
		const size_t upper_limb = round_down_to_limb(upper);

		const T _lower_mask = higher_mask(lower);
		const T _upper_mask = lower_mask(upper);

		if (lower_limb == upper_limb) {
			return (__data[upper_limb] & _lower_mask & _upper_mask) == 0u;
		}

		if (__data[lower_limb] & _lower_mask) { return false; }
		if (__data[upper_limb] & _upper_mask) { return false; }

		for (uint32_t i = lower_limb + 1u; i < upper_limb - 1u; ++i) {
			if (__data[i]) { return false; }
		}

		return true;
	}

	/// returns the position in which bits are set.
	constexpr void get_bits_set(uint32_t *P,
	                            const uint16_t pos = 1) const noexcept {
		uint16_t ctr = 0;
		for (uint32_t i = 0; i < length(); ++i) {
			if (get_bit(i)) {
				P[ctr++] = i;
				if (ctr == pos) {
					return;
				}
			}
		}
	}

	/// swap the two bits i, j
	constexpr inline void swap(const uint16_t i, const uint16_t j) noexcept {
		ASSERT(i < length() && j < length());
		auto t = get_bit_shifted(i);
		write_bit(i, get_bit_shifted(j));
		write_bit(j, t);
	}

	/// flips the bit at position `i`
	constexpr inline void flip(const uint16_t i) noexcept {
		ASSERT(i < length());
		__data[round_down_to_limb(i)] ^= mask(i);
	}

	///
	/// \tparam TT
	/// \param a
	/// \param b
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT add_T(const TT a,
	                                               const TT b) noexcept {
		return a ^ b;
	}

	///
	/// \tparam TT
	/// \param a
	/// \param b
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT sub_T(const TT a,
	                                               const TT b) noexcept {
		return a ^ b;
	}

	///
	/// \tparam TT
	/// \param a
	/// \param b
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT mul_T(const TT a,
	                                               const TT b) noexcept {
		return a & b;
	}

	///
	/// \tparam TT
	/// \param a
	/// \param b
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT scalar_T(const TT a,
	                                                  const TT b) noexcept {
		ASSERT(b < 2);
		return a * b;
	}

	///
	/// \tparam TT
	/// \param a
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT mod_T(const TT a) noexcept {
		return a;
	}

	///
	/// \tparam TT
	/// \param a
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT neg_T(const TT a) noexcept {
		return a ^ TT(-1u);
	}

	///
	/// \tparam TT
	/// \param a
	/// \return
	template<typename TT = LimbType>
	[[nodiscard]] constexpr static inline TT popcnt_T(const TT a) noexcept {
		return cryptanalysislib::popcount::popcount<TT>(a);
	}

	///
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t add256_T(const uint8x32_t a,
	                                                          const uint8x32_t b) noexcept {
		return a ^ b;
	}

	///
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t sub256_T(const uint8x32_t a,
	                                                          const uint8x32_t b) noexcept {
		return a ^ b;
	}

	///
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t mul256_T(const uint8x32_t a,
	                                                          const uint8x32_t b) noexcept {
		return a & b;
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t mod256_T(const uint8x32_t a) noexcept {
		return a;
	}

	///
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t neg256_T(const uint8x32_t a) noexcept {
		return a ^ uint8x32_t::set1(uint8_t(-1u));
	}

	///
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t scalar256_T(const uint8x32_t a,
	                                                             const uint8_t b) noexcept {
		return a * uint8x32_t::set1(b);
	}


	/// neg is an empty operation
	constexpr inline void neg(const uint16_t k_lower=0,
	                          const uint16_t k_upper=length()) noexcept {
		// do nothing.
		(void) k_lower;
		(void) k_upper;
	}

	/// full length addition.
	constexpr inline int add(BinaryContainer const &v) noexcept {
		add(*this, *this, v);
		return 0;
	}

	/// windowed addition.
	constexpr inline void add(BinaryContainer const &v,
	                          const uint32_t k_lower,
	                          const uint32_t k_upper) noexcept {
		add(*this, *this, v, k_lower, k_upper);
	}

	/// v5 = v1 ^ v2 ^ v3 ^ v4
	/// \tparam align: if set to `true` the internal simd functions will use
	///                aligned instructions.
	template<const bool align = false>
	constexpr static void add(T *v5,
	                          T const *v1,
	                          T const *v2,
	                          T const *v3,
	                          T const *v4,
	                          const uint32_t limbs) noexcept {
		uint32_t i = 0;
		constexpr uint32_t limb_size = sizeof(T);
		LOOP_UNROLL()
		for (; i + limb_size <= limbs; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<align>(v1 + i);
			uint32x8_t y_ = uint32x8_t::load<align>(v2 + i);
			uint32x8_t y1_ = uint32x8_t::load<align>(v3 + i);
			uint32x8_t y2_ = uint32x8_t::load<align>(v4 + i);
			uint32x8_t z_ = x_ ^ y_ ^ y1_ ^ y2_;
			uint32x8_t::store(v5 + i, z_);
		}

		for (; i < limbs; ++i) {
			v5[i] = v1[i] ^ v2[i] ^ v3[i] ^ v4[i];
		}
	}

	/// v4 = v1 ^ v2 ^ v3
	/// \tparam align: if set to `true` the internal simd functions will use
	///                aligned instructions.
	template<const bool align = false>
	constexpr static void add(T *v4,
	                          T const *v1,
	                          T const *v2,
	                          T const *v3,
	                          const uint32_t limbs) noexcept {
		uint64_t i = 0;
		constexpr uint32_t limb_size = sizeof(T);

		LOOP_UNROLL()
		for (; i + limb_size <= limbs; i += limb_size) {
			const uint32x8_t x_ = uint32x8_t::load<align>(v1 + i);
			const uint32x8_t y_ = uint32x8_t::load<align>(v2 + i);
			const uint32x8_t y1_ = uint32x8_t::load<align>(v3 + i);
			const uint32x8_t z_ = x_ ^ y_ ^ y1_;
			uint32x8_t::store(v4 + i, z_);
		}

		for (; i < limbs; ++i) {
			v4[i] = v1[i] ^ v2[i] ^ v3[i];
		}
	}

	/// v3 = v1 ^ v2
	/// \tparam align: if set to `true` the internal simd functions will use
	///                aligned instructions.
	template<const bool align = false>
	constexpr static inline void add(T *v3,
	                                 T const *v1,
	                                 T const *v2,
	                                 const uint32_t limbs) noexcept {
		uint64_t i = 0;
		constexpr uint32_t limb_size = sizeof(T);

		LOOP_UNROLL()
		for (; i + limb_size <= limbs; i += limb_size) {
			const uint32x8_t x_ = uint32x8_t::load<align>((uint32_t *)(v1 + i));
			const uint32x8_t y_ = uint32x8_t::load<align>((uint32_t *)(v2 + i));
			const uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store(v3 + i, z_);
		}

		for (; i < limbs; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}
	}

	/// full length addition
	constexpr static void add(T *v3,
	                          T const *v1,
	                          T const *v2) noexcept {
		return add(v3, v1, v2, limbs());
	}

	/// \param v3
	/// \param v1
	/// \param v2
	static void add_withoutasm(BinaryContainer &v3,
	                           BinaryContainer const &v1,
	                           BinaryContainer const &v2) noexcept {
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
	}

	//  IMPORTANT: this function does a full length addition
	__FORCEINLINE__ static void add(BinaryContainer &v3,
	                                BinaryContainer const &v1,
	                                BinaryContainer const &v2) noexcept {
		add(v3.ptr(), v1.ptr(), v2.ptr());
	}

	//  IMPORTANT: this function does a full length addition
	__FORCEINLINE__ static void add(BinaryContainer &v3,
									BinaryContainer const &v1,
	                                const LimbType *v2) noexcept {
		add(v3.ptr(), v1.ptr(), v2);
	}

	// add between the coordinate l, h
	template<const uint32_t k_lower, const uint32_t k_upper>
	__FORCEINLINE__ static void add(BinaryContainer &v3,
	                                BinaryContainer const &v1,
	                                BinaryContainer const &v2) noexcept {
		constexpr T lmask = higher_mask(k_lower % limb_bits_width());
		constexpr T rmask = lower_mask2(k_upper % limb_bits_width());
		constexpr uint32_t lower_limb = k_lower / limb_bits_width();
		constexpr uint32_t higher_limb = (k_upper - 1) / limb_bits_width();

		static_assert(k_lower < k_upper);
		static_assert(k_upper <= length());
		static_assert(higher_limb <= limbs());

		if constexpr (lower_limb == higher_limb) {
			constexpr T mask = k_upper % 64u == 0u ? lmask : (lmask & rmask);
			T tmp1 = (v3.__data[lower_limb] & ~(mask));
			T tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			return;
		}

		LOOP_UNROLL();
		for (uint32_t i = lower_limb + 1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		T tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		T tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		T tmp11 = (v3.__data[lower_limb] & ~(lmask));
		T tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1 ^ tmp11;
		v3.__data[higher_limb] = tmp2 ^ tmp21;
	}

	/// same as the function below.
	template<const uint32_t llimb,
	         const uint32_t ulimb,
	         const T lmask,
	         const T rmask,
	         const bool align = false>
	static void add(BinaryContainer &v3,
	                BinaryContainer const &v1,
	                BinaryContainer const &v2) noexcept {
		if constexpr (llimb == ulimb) {
			constexpr T mask = (lmask & rmask);
			T tmp1 = (v3.__data[llimb] & ~(mask));
			T tmp2 = (v1.__data[llimb] ^ v2.__data[llimb]) & mask;
			v3.__data[llimb] = tmp1 ^ tmp2;
			return;
		}

		uint32_t i = llimb + 1;

		constexpr uint32_t limb_size = sizeof(T);

		LOOP_UNROLL()
		for (; i + limb_size <= ulimb; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<align>(v1.ptr() + i);
			uint32x8_t y_ = uint32x8_t::load<align>(v2.ptr() + i);
			uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store(v3.ptr() + i, z_);
		}

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		T tmp1 = (v1.__data[llimb] ^ v2.__data[llimb]) & lmask;
		T tmp2 = (v1.__data[ulimb] ^ v2.__data[ulimb]) & rmask;
		T tmp11 = (v3.__data[llimb] & ~(lmask));
		T tmp21 = (v3.__data[ulimb] & ~(rmask));

		v3.__data[llimb] = tmp1 ^ tmp11;
		v3.__data[ulimb] = tmp2 ^ tmp21;
	}


	/// Another heavy overloaded function to add two vectors. Add two vector v1+v3=v3 and safe the result in v3.
	/// But only calculates the sum between the limbs `llimb` and `ulimb` while apply to these limbs the masks
	/// \tparam llimb 	lowest limb
	/// \tparam ulimb 	highest limb
	/// \tparam lmask	bit mask for llimb
	/// \tparam rmask	bit mask for ulimb
	/// \return nothing
	template<const uint32_t llimb,
	         const uint32_t ulimb,
	         const T lmask,
	         const T rmask,
	         const bool align = false>
	constexpr static void add(T *v3,
	                          T const *v1,
	                          T const *v2) noexcept {
		if constexpr (llimb == ulimb) {
			constexpr T mask = (lmask & rmask);
			T tmp1 = (v3[llimb] & ~(mask));
			T tmp2 = (v1[llimb] ^ v2[llimb]) & mask;
			v3[llimb] = tmp1 ^ tmp2;
			return;
		}

		int32_t i = llimb + 1;

		constexpr uint32_t limb_size = sizeof(T);

		LOOP_UNROLL()
		for (; i + limb_size <= ulimb; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<align>(v1 + i);
			uint32x8_t y_ = uint32x8_t::load<align>(v2 + i);
			uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store(v3 + i, z_);
		}

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}

		T tmp1 = (v1[llimb] ^ v2[llimb]) & lmask;
		T tmp2 = (v1[ulimb] ^ v2[ulimb]) & rmask;
		T tmp11 = (v3[llimb] & ~(lmask));
		T tmp21 = (v3[ulimb] & ~(rmask));

		v3[llimb] = tmp1 ^ tmp11;
		v3[ulimb] = tmp2 ^ tmp21;
	}

	/// Does a full length addition. But the hamming weight is just calculated up to `ulimb` with the mask `rmask`
	/// \tparam ulimb	max limb to calc the hamming weight
	/// \tparam rmask	mask to apply before calc the weight.
	/// \return
	template<const uint32_t ulimb,
	         const T rmask>
	inline static uint32_t add_only_upper_weight_partly(
	        BinaryContainer &v3,
	        BinaryContainer const &v1,
	        BinaryContainer const &v2) noexcept {
		int32_t i = 0;
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcnt_T(v3.__data[i]);
		}

		v3.__data[ulimb] = v1.__data[ulimb] ^ v2.__data[ulimb];
		return hm + popcnt_T(v3.__data[ulimb] & rmask);
	}

	///
	/// \tparam ulimb
	/// \tparam rmask
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	template<const uint32_t ulimb,
	         const T rmask>
	inline static uint32_t add_only_upper_weight_partly_withoutasm(
	        BinaryContainer &v3,
	        BinaryContainer const &v1,
	        BinaryContainer const &v2) noexcept {
		uint32_t i = 0;
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcnt_T(v3.__data[i]);
		}

		for (; i < limbs(); i++) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
		return hm + popcnt_T(v3.__data[ulimb] & rmask);
	}

	///
	/// \tparam j
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	template<const uint32_t j>
	inline static uint32_t add_only_upper_weight_partly_withoutasm(
	        BinaryContainer &v3,
	        BinaryContainer const &v1,
	        BinaryContainer const &v2) noexcept {
		constexpr uint32_t ulimb = round_down_to_limb(j - 1);
		constexpr static T rmask = lower_mask2(j);

		int32_t i = 0;
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcnt_T(v3.__data[i]);
		}

		for (; i < limbs(); i++) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}
		return hm + popcnt_T(v3.__data[ulimb] & rmask);
	}

	///
	/// \tparam ulimb
	/// \tparam rmask
	/// \tparam early_exit
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	template<const uint32_t ulimb,
	         const T rmask,
	         const uint32_t early_exit>
	static uint32_t add_only_upper_weight_partly_withoutasm_earlyexit(
	        BinaryContainer &v3,
	        BinaryContainer const &v1,
	        BinaryContainer const &v2) noexcept {
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (uint32_t i = 0; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcnt_T(v3.__data[i]);
			if (hm > early_exit)
				return hm;
		}

		v3.__data[ulimb] = v1.__data[ulimb] ^ v2.__data[ulimb];
		return hm + popcnt_T(v3.__data[ulimb] & rmask);
	}

	///
	/// \tparam ulimb
	/// \tparam rmask
	/// \tparam early_exit
	/// \tparam align
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	template<const uint32_t ulimb,
	         const T rmask,
	         const uint32_t early_exit,
	         const bool align = false>
	static uint32_t add_only_upper_weight_partly_earlyexit(
	        BinaryContainer &v3,
	        BinaryContainer const &v1,
	        BinaryContainer const &v2) noexcept {
		uint32_t hm = 0;
		uint32_t i = 0;

		constexpr uint32_t limb_size = sizeof(T);

		LOOP_UNROLL()
		for (; i + limb_size <= ulimb; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<align>(v1.ptr() + i);
			uint32x8_t y_ = uint32x8_t::load<align>(v2.ptr() + i);
			uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store(v3.ptr() + i, z_);
		}

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		LOOP_UNROLL();
		for (uint32_t i = 0; i < ulimb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			hm += popcnt_T(v3.__data[i]);
			if (hm > early_exit)
				return hm;
		}

		v3.__data[ulimb] = v1.__data[ulimb] ^ v2.__data[ulimb];
		return hm + popcnt_T(v3.__data[ulimb] & rmask);
	}

	/// calculates the sumf of v3= v1+v2 on the full length, but return the weight of v3 only ont the first coordinate
	/// defined by `ulimb` and `rmask`
	/// \tparam ulimb	max limb to calculate the weight on the full length of each limb
	/// \tparam rmask	mask which cancel out unwanted bits on the last
	/// \return hamming weight
	template<const uint32_t ulimb, const T rmask>
	static uint32_t add_only_upper_weight_partly(T *v3, T const *v1, T const *v2) noexcept {
		int32_t i = 0;
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3[i] = v1[i] ^ v2[i];
			hm += popcnt_T(v3[i]);
		}

		v3[ulimb] = v1[ulimb] ^ v2[ulimb];
		return hm + popcnt_T(v3[ulimb] & rmask);
	}

	/// calculates the sum of v3= v1+v2 on the partly length
	/// \tparam ulimb	max limb to calculate the weight on the full length of each limb
	/// \tparam rmask	mask which cancel out unwanted bits on the last
	template<const uint32_t ulimb,
	         const T rmask,
	         const bool align = false>
	constexpr static void add_only_upper(T *v3,
	                                     T const *v1,
	                                     T const *v2) noexcept {
		if constexpr (0 == ulimb) {
			v3[0] = (v1[0] ^ v2[0]) & rmask;
			return;
		}

		int32_t i = 0;

		constexpr uint32_t limb_size = sizeof(T);

		LOOP_UNROLL()
		for (; i + limb_size <= ulimb; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<align>(v1 + i);
			uint32x8_t y_ = uint32x8_t::load<align>(v2 + i);
			uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store(v3 + i, z_);
		}

		LOOP_UNROLL();
		for (; i < ulimb; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}

		v3[ulimb] = (v1[i] ^ v2[i]) & rmask;
	}

	constexpr static void add(BinaryContainer &v3,
	                          BinaryContainer const &v1,
	                          BinaryContainer const &v2,
	                          const uint32_t k_lower,
	                          const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		const T lmask = higher_mask(k_lower % limb_bits_width());
		const T rmask = lower_mask2(k_upper % limb_bits_width());
		const int64_t lower_limb = k_lower / limb_bits_width();
		const int64_t higher_limb = (k_upper - 1) / limb_bits_width();

		if (lower_limb == higher_limb) {
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			T tmp1 = (v3.__data[lower_limb] & ~(mask));
			T tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			return;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb + 1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
		}

		T tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		T tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		T tmp11 = (v3.__data[lower_limb] & ~(lmask));
		T tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1 ^ tmp11;
		v3.__data[higher_limb] = tmp2 ^ tmp21;
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr static uint32_t add_weight(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		static_assert(k_upper <= length() && k_lower < k_upper && 0 < k_upper);

		uint32_t cnorm = 0;
		constexpr T lmask = higher_mask(k_lower % limb_bits_width());
		constexpr T rmask = lower_mask2(k_upper % limb_bits_width());
		constexpr uint32_t lower_limb = k_lower / limb_bits_width();
		constexpr uint32_t higher_limb = (k_upper - 1) / limb_bits_width();

		if constexpr (lower_limb == higher_limb) {
			constexpr T mask = k_upper % limb_bits_width() == 0 ? lmask : (lmask & rmask);
			T tmp1 = (v3.__data[lower_limb] & ~(mask));
			T tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			auto b = popcnt_T(tmp2);
			return b;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb + 1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcnt_T(v3.__data[i]);
		}

		T tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		T tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		T tmp11 = (v3.__data[lower_limb] & ~(lmask));
		T tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1 ^ tmp11;
		v3.__data[higher_limb] = tmp2 ^ tmp21;

		cnorm += popcnt_T(tmp1);
		cnorm += popcnt_T(tmp2);

		return cnorm;
	}

	constexpr static uint32_t add_weight(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2,
	                                     const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper && 0 < k_upper);

		uint32_t cnorm = 0;
		const T lmask = higher_mask(k_lower % limb_bits_width());
		const T rmask = lower_mask2(k_upper % limb_bits_width());
		const int64_t lower_limb = k_lower / limb_bits_width();
		const int64_t higher_limb = (k_upper - 1) / limb_bits_width();

		if (lower_limb == higher_limb) {
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			T tmp1 = (v3.__data[lower_limb] & ~(mask));
			T tmp2 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & mask;
			v3.__data[lower_limb] = tmp1 ^ tmp2;
			auto b = popcnt_T(tmp2);
			return b;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb + 1; i < higher_limb; ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcnt_T(v3.__data[i]);
		}

		T tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		T tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		T tmp11 = (v3.__data[lower_limb] & ~(lmask));
		T tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1 ^ tmp11;
		v3.__data[higher_limb] = tmp2 ^ tmp21;

		cnorm += popcnt_T(tmp1);
		cnorm += popcnt_T(tmp2);

		return cnorm;
	}

	inline constexpr static bool add(BinaryContainer &v3,
	                                 BinaryContainer const &v1,
	                                 BinaryContainer const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper,
	                                 const uint32_t norm) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);

		if (norm == uint32_t(-1)) {
			// fallback to normal addition.
			add(v3, v1, v2, k_lower, k_upper);
			return false;
		} else {
			ASSERT((&v1 != &v3 && &v2 != &v3) || (norm == uint32_t(-1)));
			uint32_t cnorm = add_weight(v3, v1, v2, k_lower, k_upper);
			if (cnorm >= norm)
				return true;
		}

		return false;
	}

	// calculates the sum v3=v1+v2 and returns the hamming weight of v3
	inline constexpr static uint32_t add_weight(T *v3, T const *v1, T const *v2) noexcept {
		uint32_t cnorm = 0;
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3[i] = v1[i] ^ v2[i];
			cnorm += popcnt_T(v3[i]);
		}

		return cnorm;
	}

	inline constexpr static uint32_t add_weight(BinaryContainer &v3, BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		uint32_t cnorm = 0;
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3.__data[i] = v1.__data[i] ^ v2.__data[i];
			cnorm += popcnt_T(v3.__data[i]);
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
	inline constexpr static int sub(BinaryContainer &v3,
	                                BinaryContainer const &v1,
	                                BinaryContainer const &v2) noexcept {
		add(v3, v1, v2);
		return 0; // always return it's ok and doesn't need to be filtered
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
		return cmp(v1, v2, 0, length());
	}

	/// implements only a 2 way comparison. E.g. implements the `!=` operator.
	inline constexpr static bool cmp(BinaryContainer const &v1, BinaryContainer const &v2,
	                                 const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		const int32_t lower = round_down_to_limb(k_lower);
		const int32_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T rmask = lower_mask2(k_upper);

		if (lower == upper) {// the two offsets lay in the same limb.
			const T mask = k_upper % limb_bits_width() == 0 ? lmask : (lmask & rmask);
			return cmp_simple2(v1, v2, lower, mask);
		} else {// the two offsets lay in two different limbs
			// first check the highest limb with the mask
			return cmp_ext2(v1, v2, lower, upper, lmask, rmask);
		}
	}

	/// Important: lower != higher
	/// unrolled high speed implementation of a multi limb compare function
	template<const uint32_t lower, const uint32_t upper, const T lmask, const T umask>
	inline constexpr static bool cmp_ext(BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		ASSERT(lower != upper && lower < upper);

		// first check the highest limb with the mask
		if ((v1.__data[upper] & umask) != (v2.__data[upper] & umask))
			return false;

		// check all limbs in the middle
		LOOP_UNROLL()
		for (uint64_t i = upper - 1; i > lower; i--) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		if ((v1.__data[lower] & lmask) != (v2.__data[lower] & lmask))
			return false;
		return true;
	}

	/// IMPORTANT; lower < upper so you have to compare at least two limbs.
	/// use this function if you have to compare a lot of different elements on the same coordinate. So you can precompute
	/// the mask and the limbs.
	inline constexpr static bool cmp_ext2(BinaryContainer const &v1, BinaryContainer const &v2,
	                                      const uint32_t lower, const uint32_t upper, const T lmask, const T umask) noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// first check the highest limb with the mask.
		if ((v1.__data[upper] & umask) != (v2.__data[upper] & umask))
			return false;

		// check all limbs in the middle.
		for (uint64_t i = upper - 1; i > lower; i--) {
			if (v1.__data[i] != v2.__data[i])
				return false;
		}

		// and at the end check the lowest limb.
		if ((v1.__data[lower] & lmask) != (v2.__data[lower] & lmask))
			return false;
		return true;
	}

	/// IMPORTANT: lower != higher => mask != 0. This is actually only a sanity check.
	/// high speed implementation of a same limb cmompare function
	template<const uint32_t limb, const T mask>
	inline constexpr static bool cmp_simple(BinaryContainer const &v1, BinaryContainer const &v2) noexcept {
		ASSERT(limb != uint64_t(-1) && mask != 0);
		return ((v1.__data[limb] & mask) == (v2.__data[limb] & mask));
	}

	/// IMPORTANT: mask != 0.
	/// use this function if you have to compare a lot of different elements on the same coordinate. So you can precompute
	/// the mask and the limb.
	inline constexpr static bool cmp_simple2(BinaryContainer const &v1, BinaryContainer const &v2, const uint32_t limb, const T mask) noexcept {
		ASSERT(limb != uint32_t(-1) && mask != 0);
		return ((v1.__data[limb] & mask) == (v2.__data[limb] & mask));
	}

	inline constexpr static int cmp_ternary_simple2(BinaryContainer const &v1, BinaryContainer const &v2, const uint32_t limb, const T mask) noexcept {
		ASSERT(limb != uint64_t(-1));
		if ((v1.__data[limb] & mask) > (v2.__data[limb] & mask))
			return 1;
		else if ((v1.__data[limb] & mask) < (v2.__data[limb] & mask))
			return -1;

		return 0;
	}

	/// IMPORTANT: k_lower < k_upper is enforced.
	/// sets v1 = v2[k_lower, ..., k_upper].
	/// Does not change anything else.
	inline constexpr static void set(BinaryContainer &v1, BinaryContainer const &v2,
	                                 const uint32_t k_lower, const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		const int64_t lower = round_down_to_limb(k_lower);
		const int64_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T rmask = lower_mask2(k_upper);

		if (lower == upper) {// the two offsets lay in the same limb.
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			v1.__data[lower] = (v1.__data[lower] & ~mask) | (v2.__data[lower] & mask);
			return;
		} else {// the two offsets lay in two different limbs
			v1.__data[upper] = (v1.__data[upper] & ~rmask) | (v2.__data[upper] & rmask);
			v1.__data[lower] = (v1.__data[lower] & ~lmask) | (v2.__data[lower] & lmask);
			for (uint32_t i = upper - 1ul; (i > lower) && (upper > 0); i--) {
				v1.__data[i] = v2.__data[i];
			}
		}
	}

	// TODO missing rol/ror
	///  out[s: ] = in[0:s]
	constexpr static inline void sll(BinaryContainer &out,
	                               const BinaryContainer &in,
	                               const uint32_t s) noexcept {
		out.zero();

		ASSERT(s < length());
		for (uint32_t j = 0; j < length() - s; ++j) {
			const auto bit = in.get_bit_shifted(j);
			out.write_bit(j + s, bit);
		}
	}

	///  out[0: ] = in[s:]
	constexpr static inline void slr(BinaryContainer &out,
						   const BinaryContainer &in,
						   const uint32_t s) noexcept {
		ASSERT(s < length());
		for (uint32_t j = s; j < length(); ++j) {
			out.write_bit(j-s, in.get_bit_shifted(j));
		}
		for (uint32_t j = s; j < length(); ++j) {
			out.write_bit(j, 0);
		}
	}

	/// checks whether this == obj on the interval [k_lower, ..., k_upper]
	/// the level of the calling 'list' object.
	/// \return
	inline bool is_equal(const BinaryContainer &obj,
	                     const uint32_t k_lower = 0,
	                     const uint32_t k_upper = length()) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_equal(const BinaryContainer &obj) const noexcept {
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper - 1);
		constexpr T lmask = higher_mask(k_lower);
		constexpr T rmask = lower_mask2(k_upper);

		if constexpr (lower == upper) {
			constexpr T mask = lmask & rmask;
			return (__data[lower] ^ mask) == (obj.__data[lower] ^ mask);
		}

		return cmp_ext<lower, upper, lmask, rmask>(*this, obj);
	}

	template<const uint32_t lower, const uint32_t upper, const T lmask, const T umask>
	inline bool is_equal_ext(const BinaryContainer &obj) const noexcept {
		return cmp_ext2<lower, upper, lmask, umask>(*this, obj);
	}

	inline bool is_equal_ext2(const BinaryContainer &obj, const uint32_t lower, const uint32_t upper,
	                          const T lmask, const T umask) const noexcept {
		return cmp_ext2(*this, obj, lower, upper, lmask, umask);
	}

	inline bool is_equal_simple2(const BinaryContainer &obj, const uint32_t limb, const T mask) const noexcept {
		return cmp_simple2(*this, obj, limb, mask);
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_greater(BinaryContainer const &obj) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper - 1);
		constexpr T lmask = higher_mask(k_lower);
		constexpr T rmask = lower_mask2(k_upper);
		if constexpr (lower == upper) {// the two offsets lay in the same limb.
			constexpr T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			return ((__data[lower] & mask) > (obj.__data[lower] & mask));
		} else {// the two offsets lay in two different limbs
			ASSERT(0);
			return 0;
		}
	}

	/// implements a strict comparison. Call this function if you dont know what to call. Its the most generic implementaion
	/// and it works for all input.s
	inline bool is_greater(BinaryContainer const &obj,
	                       const uint32_t k_lower = 0,
	                       const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		int64_t lower = round_down_to_limb(k_lower);
		int64_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T rmask = lower_mask2(k_upper);

		if (lower == upper) {// the two offsets lay in the same limb.
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			return this->is_greater_simple2(obj, lower, mask);
		} else {// the two offsets lay in two different limbs
			return this->is_greater_ext2(obj, lower, upper, lmask, rmask);
		}
	}

	/// return *this > obj on the limbs [lower, upper]. `lmask` and `umask` are bitmask for the lowest and highest limbs.
	/// Technically this is a specially unrolled implementation of `BinaryContainer::is_greater` if you have to compare a lot
	/// of containers repeatedly on the same coordinates.
	inline bool is_greater_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                            const T lmask, const T umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldn't make sense.

		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, >, <)
		// check all limbs in the middle
		for (uint64_t i = upper - 1; i > lower; i--) {
			BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], >, <)
		}

		BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, >, <)
		return false;
	}

	inline bool is_greater_equal_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                                  const T lmask, const T umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldn't make sense.
		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, >=, <)
		// check all limbs in the middle
		for (uint64_t i = upper - 1; i > lower; i--) {
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
	inline bool is_greater_simple2(BinaryContainer const &obj, const uint32_t limb, const T mask) const noexcept {
		ASSERT(limb < limbs() && mask != 0);
		return ((__data[limb] & mask) > (obj.__data[limb] & mask));
	}

	/// not testet
	inline bool is_greater_equal_simple2(BinaryContainer const &obj, const uint32_t limb, const T mask) const noexcept {
		ASSERT(limb < limbs() && mask != 0);
		return ((__data[limb] & mask) >= (obj.__data[limb] & mask));
	}

	/// main comparison function for the < operator. If you dont know what function to use, use this one. Its the most generic
	/// implementation and works for all inputs.
	inline bool is_lower(BinaryContainer const &obj,
	                     const uint32_t k_lower = 0,
	                     const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		int64_t lower = round_down_to_limb(k_lower);
		int64_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T rmask = lower_mask2(k_upper);

		if (lower == upper) {// the two offsets lay in the same limb.
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			return is_lower_simple2(obj, lower, mask);
		} else {// the two offsets lay in two different limbs
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
	///		const BinaryContainerTest2::T lmask = BinaryContainerTest2::higher_mask(k_higher);
	///		const BinaryContainerTest2::T umask = BinaryContainerTest2::lower_mask2(b1.size());
	///		is_lower_ext2(b2, lower, upper, lmask, umask))
	/// You __MUST__ be extremely carefully with the chose of `upper`.
	/// Normally you want to compare two elements between `k_lower` and `k_upper`. This can be done by:
	///		const uint64_t lower = BinaryContainerTest2::round_down_to_limb(k_lower);
	///		const uint64_t upper = BinaryContainerTest2::round_down_to_limb(k_higher);
	///							... (as above)
	/// Note that you dont have to pass the -1 to k_higher. The rule of thumb is that you __MUST__ add a -1 to the computation
	/// of the upper limb.
	inline bool is_lower_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                          const T lmask, const T umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise, cases like k_upper = 128 wouldn't make sense.

		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, <, >)
		// check all limbs in the middle
		for (uint64_t i = upper - 1; i > lower; i--) {
			BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], <, >)
		}

		BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, <, >)
		return false;
	}

	/// not testet
	inline bool is_lower_equal_ext2(BinaryContainer const &obj, const uint32_t lower, const uint32_t upper,
	                                const T lmask, const T umask) const noexcept {
		ASSERT(lower < upper && lmask != 0 && upper < limbs());
		// umask is allowed to be zero. Otherwise cases like k_upper = 128 wouldnt make sense.

		BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, <=, >)
		// check all limbs in the middle
		for (uint64_t i = upper - 1; i > lower; i--) {
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
	inline bool is_lower_simple2(BinaryContainer const &obj,
	                             const uint32_t limb,
	                             const T mask) const noexcept {
		ASSERT(limb < limbs());
		ASSERT(limbs() < length());
		ASSERT(mask != 0);
		return ((__data[limb] & mask) < (obj.__data[limb] & mask));
	}

	inline bool is_lower_equal_simple2(BinaryContainer const &obj, const uint32_t limb, const T mask) const noexcept {
		ASSERT((limb < limbs() < length()) && mask != 0);
		return ((__data[limb] & mask) <= (obj.__data[limb] & mask));
	}

	// calcs the weight up to (include) ilumb at early exits if its bigger than early exit.
	template<const uint32_t ulimb, const T rmask, const uint32_t early_exit>
	inline uint32_t weight_earlyexit() noexcept {
		uint32_t hm = 0;

		LOOP_UNROLL();
		for (uint32_t i = 0; i < ulimb; ++i) {
			hm += popcnt_T(__data[i]);
			if (hm > early_exit)
				return hm;
		}

		return hm + popcnt_T(__data[ulimb] & rmask);
	}

	/// TODO add to concept and add to kAryTypes
	/// NOTE: need to abs
	constexpr static inline uint32_t dist(const BinaryContainer &a,
	                                      const BinaryContainer &b) noexcept {
		uint32_t ret = 0;
		// TODO simd
		for (uint32_t i = 0; i < limbs(); ++i) {
			ret += popcnt_T(a.__data[i] ^ b.__data[i]);
		}

		return ret;
	}

	uint32_t popcnt(const uint32_t k_lower = 0, const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		const uint32_t lower = round_down_to_limb(k_lower);
		const uint32_t upper = round_down_to_limb(k_upper - 1);

		const T l_mask = higher_mask(k_lower);
		const T u_mask = lower_mask2(k_upper);

		uint32_t weight;
		// if only one limb needs to be checked to check
		if (lower == upper) {
			uint64_t b = (l_mask & u_mask);
			uint64_t c = uint64_t(__data[lower]);
			uint64_t d = uint64_t(b) & uint64_t(c);
			uint64_t w_ = popcnt_T(d);
			return w_;
		}

		weight = popcnt_T(l_mask & __data[lower]);
		weight += popcnt_T(u_mask & __data[upper]);
		for (uint32_t i = lower + 1; i < upper; ++i) {
			weight += popcnt_T(__data[i]);
		}

		return weight;
	}

	template<const uint32_t lower, const uint32_t upper, const T l_mask, const T u_mask>
	constexpr uint32_t popcnt() const noexcept {
		ASSERT(lower <= upper);
		uint32_t weight = 0;

		// if only one limb needs to be checked to check
		if constexpr (lower == upper) {
			uint64_t b = (l_mask & u_mask);
			uint64_t c = uint64_t(__data[lower]);
			uint64_t d = uint64_t(b) & uint64_t(c);
			uint64_t w_ = popcnt_T(d);
			return w_;
		}

		weight = popcnt_T(l_mask & __data[lower]);
		for (uint32_t i = lower + 1; i < upper; ++i)
			weight += popcnt_T(__data[i]);
		weight += popcnt_T(u_mask & __data[upper]);
		return weight;
	}

	/// Calculates a+=b on full length while caluclating the hamming wight only on the first coordinates, which are
	/// specified by `upper` and `u_mask`
	/// \tparam upper	limb position of the last normal addition
	/// \tparam u_mask	mask to apply the calculate the weight on the last limb
	/// \param a	input vector
	/// \param b	intpur vector
	/// \return	hamming weight of the first coordinates.
	template<const uint32_t upper, const T u_mask>
	constexpr static uint32_t weight_sum_only_upper(T *a, T *b) noexcept {
		uint32_t weight = 0;

		// if only one limb needs to be checked to check
		if constexpr (0 == upper) {
			T c = T(a[0] ^ b[0]);
			T d = T(u_mask) & T(c);
			T w_ = popcnt_T(d);
			return w_;
		}

		for (uint32_t i = 0; i < upper; ++i)
			weight += popcnt_T(a[i] ^ b[i]);

		return weight + popcnt_T(u_mask & (a[upper] ^ b[upper]));
	}

	// hack it like
	class reference {
		friend class BinaryContainer;

		// pointer to the limb
		T *wp;

		// bit position in the whole data const_array.
		const size_t mask_pos;

		// left undefined
		reference();

	public:
		constexpr reference(const BinaryContainer &b, const size_t pos) : mask_pos(mask(pos)) {
			// honestly thats cheating. We drop the const qualifier here, s.t.
			// we can get a const reference
			wp = (T *) &b.data().data()[round_down_to_limb(pos)];
		}

#if __cplusplus >= 201103L
		reference(const reference &) = default;
#endif

		constexpr ~reference() = default;

		// For b[i] = __x;
		reference &operator=(bool x) {
			if (x)
				*wp |= mask_pos;
			else
				*wp &= ~mask_pos;
			return *this;
		}

		// For b[i] = b[__j];
		[[nodiscard]]constexpr reference &operator=(const reference &j) noexcept {
			if (*(j.wp) & j.mask_pos) {
				*wp |= mask_pos;
			} else {
				*wp &= ~mask_pos;
			}
			return *this;
		}

		// Flips the bit
		[[nodiscard]] bool operator~() const noexcept { return (*(wp) &mask_pos) == 0; }

		// For __x = b[i];
		[[nodiscard]] constexpr operator bool() const noexcept {
			return (*(wp) &mask_pos) != 0;
		}

		// For b[i].flip();
		[[nodiscard]] constexpr reference &flip() noexcept {
			*wp ^= mask_pos;
			return *this;
		}

		[[nodiscard]] constexpr inline unsigned int get_data() const noexcept { return bool(); }
		[[nodiscard]] constexpr inline unsigned int data() const noexcept { return bool(); }
	};
	friend class reference;


	/// access operators
	constexpr void set(const bool data) noexcept {
		for (uint32_t i = 0; i < length(); i++) {
			set(data, i);
		}
	}

	///
	/// \param data
	/// \param pos
	/// \return
	constexpr void set(const bool data,
	                   const size_t pos) noexcept {
		ASSERT(pos < length());
		reference(*this, pos) = data;
	}
	[[nodiscard]] constexpr reference get(const size_t pos) noexcept {
		ASSERT(pos < length());
		return reference(*this, pos);
	}
	[[nodiscard]] constexpr const reference get(const size_t pos) const noexcept {
		ASSERT(pos < length());
		return (const reference) reference(*this, pos);
	}
	[[nodiscard]] constexpr reference operator[](const size_t pos) noexcept {
		ASSERT(pos < length());
		return reference(*this, pos);
	}
	[[nodiscard]] constexpr bool operator[](const size_t pos) const noexcept {
		ASSERT(pos < length());
		return (__data[round_down_to_limb(pos)] & mask(pos)) != 0;
	}

	/// Assignment operator implementing copy assignment
	/// see https://en.cppreference.com/w/cpp/language/operators
	/// \param obj
	/// \return
	BinaryContainer &operator=(BinaryContainer const &obj) noexcept {
		if (this != &obj) {// self-assignment check expected
			std::copy(&obj.__data[0], &obj.__data[0] + obj.__data.size(), &this->__data[0]);
		}

		return *this;
	}

	/// Assignment operator implementing move assignment
	/// Alternative definition: Value& operator =(Value &&obj) = default;
	/// see https://en.cppreference.com/w/cpp/language/move_assignment
	/// \param obj
	/// \return
	BinaryContainer &operator=(BinaryContainer &&obj) noexcept {
		if (this != &obj) {// self-assignment check expected really?
			// move the data
			__data = std::move(obj.__data);
		}

		return *this;
	}

	/// wrapper around `print`
	void print_binary(const uint32_t k_lower = 0, const uint32_t k_upper = length()) const noexcept {
		print(k_lower, k_upper);
	}

	/// print some information
	/// \param k_lower lower limit to print (included)
	/// \param k_upper higher limit to print (not included)
	void print(const uint32_t k_lower = 0, const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_lower < length() && k_upper <= length() && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << data(i) << "";
		}
		std::cout << "\n";
	}

	//T data(uint64_t index) { ASSERT(index < length); return get_bit_shifted(index); }
	bool data(uint64_t index) const noexcept {
		ASSERT(index < length());
		return get_bit_shifted(index);
	}

	// simple hash function
	constexpr inline uint64_t hash() const noexcept {
		return __data[0];
	}

#ifdef BINARY_CONTAINER_ALIGNMENT
	static constexpr uint16_t alignment() {
		// Aligns to a multiple of 32 Bytes
		//constexpr uint16_t bytes = (length+7)/8;
		//constexpr uint16_t limbs = (((bytes-1) >> 5)+1)<<5;
		//return limbs*8;

		// Aligns to a multiple of 16 Bytes
		constexpr uint16_t bytes = (length + 7) / 8;
		constexpr uint16_t limbs = (((bytes - 1) >> 4) + 1) << 4;
		return limbs * 8;
	}
#endif

	// length operators
	__FORCEINLINE__ constexpr static bool binary() noexcept { return true; }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return length(); }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return (length() + limb_bits_width() - 1) / limb_bits_width(); }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept {
#ifdef BINARY_CONTAINER_ALIGNMENT
		return alignment() / 8;
#else
		return limbs() * sizeof(T);
#endif
	}

	__FORCEINLINE__ T *ptr() noexcept { return __data.data(); };
	const __FORCEINLINE__ T *ptr() const noexcept { return __data.data(); };
	__FORCEINLINE__ T ptr(const size_t i) noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};
	const __FORCEINLINE__ T ptr(const size_t i) const noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};

	// get the internal data vector
	__FORCEINLINE__ std::array<T, compute_limbs()> &data() noexcept { return __data; };
	__FORCEINLINE__ const std::array<T, compute_limbs()> &data() const noexcept { return __data; };

	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	__FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };

	///
	constexpr static void info() noexcept {
		std::cout << "{ name: \"kAryContainerMeta\""
				  << ", n: " << n
				  << ", q: " << q
				  << ", sizeof(T): " << sizeof(T)
				  << "}" << std::endl;
	}
private:
	// actual data container.
	std::array<T, compute_limbs()> __data;
};


template<uint64_t _n,
        typename T=uint64_t>
std::ostream &operator<<(std::ostream &out,
                         const BinaryContainer<_n, T> &obj) {
	for (size_t i = 0; i < obj.length(); ++i) {
		out << obj[i];
	}
	return out;
}

#endif//SMALLSECRETLWE_CONTAINER_H
