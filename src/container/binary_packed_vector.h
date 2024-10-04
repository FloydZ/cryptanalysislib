#ifndef CRYPTANALYSISLIB_BINARYPACKEDCONTAINER_H
#define CRYPTANALYSISLIB_BINARYPACKEDCONTAINER_H

#include <array>
#include <atomic>
#include <cstdint>

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

// local includes
#include "container/fq_packed_vector.h"
#include "helper.h"
#include "popcount/popcount.h"
#include "random.h"
#include "simd/simd.h"
#include "hash/hash.h"

// TODO alle funktionen gleich anordnen wie in der fq klasse

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
template<const uint32_t _n,
		 typename T>
#if __cplusplus > 201709L
    requires std::is_arithmetic_v<T>
#endif
class FqPackedVector<_n, 2, T> : public FqPackedVectorMeta<_n, 2, T> {
public:
	// Internal Types needed for the template system.
	typedef FqPackedVector<_n, 2, T> ContainerType;
	typedef T LimbType;
	typedef bool DataType;
	using S = uint8x32_t;

	// internal data length. Need to export it for the template system.
	constexpr static uint32_t n = _n;
	[[nodiscard]] constexpr static inline uint32_t length() noexcept { return n; }
	constexpr static uint64_t q = 2;
	[[nodiscard]] constexpr static inline uint64_t modulus() noexcept { return q; }

	constexpr static uint32_t RADIX = sizeof(T) * 8;
	constexpr static T minus_one = T(-1);

	static_assert(_n > 0);

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

	/// default constructor
	constexpr FqPackedVector() noexcept : __data() {}

	/// Copy Constructor
	constexpr FqPackedVector(const FqPackedVector &a) noexcept : __data(a.__data) {}

	/// Assignment operator implementing copy assignment
	/// see https://en.cppreference.com/w/cpp/language/operators
	/// \param obj
	/// \return
	constexpr FqPackedVector &operator=(FqPackedVector const &obj) noexcept {
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
	constexpr FqPackedVector &operator=(FqPackedVector &&obj) noexcept {
		if (this != &obj) {// self-assignment check expected really?
			// move the data
			__data = std::move(obj.__data);
		}

		return *this;
	}


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

	/// set the whole data const_array on rng data.
	void random() noexcept {
		constexpr uint64_t apply_mask = length() % limb_bits_width() == 0 ? lower_mask(length()) - 1 : lower_mask(length());

		if constexpr (length() < 64) {
			__data[0] = rng() & apply_mask;
		} else {
			for (uint32_t i = 0; i < limbs() - 1; ++i) {
				__data[i] = rng();
			}
			__data[limbs() - 1] = rng() & apply_mask;
		}
	}


	void random(const uint32_t lower,
	            const uint32_t upper) noexcept {
		ASSERT(lower < upper);
		ASSERT(upper <= length());

		const size_t lower_limb = round_down_to_limb(lower);
		const size_t upper_limb = round_down_to_limb(upper);

		const T _lower_mask = higher_mask(lower);
		const T _upper_mask = lower_mask(upper);

		if (lower_limb == upper_limb) {
			const T mask =  _lower_mask & _upper_mask;
			__data[lower_limb] ^= rng<T>() & mask;
		}

		__data[lower_limb] ^= rng<T>() & _lower_mask;
		__data[upper_limb] ^= rng<T>() & _upper_mask;

		for (uint32_t i = lower_limb + 1u; i < upper_limb - 1u; ++i) {
			__data[lower_limb] ^= rng<T>();
		}
	}

	/// split the full length BinaryContainer into `k` windows.
	/// Inject in every window weight `w` on rng positions.
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
				uint64_t pos = rng() % (windows_length - l);
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
			uint64_t pos = rng() % (m - i);
			bool t = get_bit_shifted(i+offset);
			write_bit(i+offset, get_bit_shifted(i + pos + offset));
			write_bit(i + pos + offset, t);
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

	/// \param lower lower limit, inclusive
	/// \param upper upper limit, exclusive
	/// \return whether the vector is zero between [k_lower, k_upper)
	template<const uint32_t lower, uint32_t upper>
	[[nodiscard]] constexpr inline bool is_zero()const noexcept {
		ASSERT(upper <= length());
		constexpr size_t lower_limb = round_down_to_limb(lower);
		constexpr size_t upper_limb = round_down_to_limb(upper);
		constexpr T _lower_mask = higher_mask(lower);
		constexpr T _upper_mask = lower_mask(upper);

		if constexpr (lower_limb == upper_limb) {
			return (__data[upper_limb] & _lower_mask & _upper_mask) == 0u;
		}

		if (__data[lower_limb] & _lower_mask) { return false; }
		if (__data[upper_limb] & _upper_mask) { return false; }

		for (uint32_t i = lower_limb + 1u; i < upper_limb - 1u; ++i) {
			if (__data[i]) { return false; }
		}

		return true;
	}


	/// implements only a 2 way comparison. E.g. implements the `!=` operator.
	[[nodiscard]] inline constexpr static bool cmp(FqPackedVector const &v1,
	                                               FqPackedVector const &v2,
									               const uint32_t k_lower=0,
	                                               const uint32_t k_upper=length()) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		const int32_t lower = round_down_to_limb(k_lower);
		const int32_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T umask = lower_mask2(k_upper);

		if (lower == upper) {// the two offsets lay in the same limb.
			const T mask = k_upper % limb_bits_width() == 0 ? lmask : (lmask & umask);
			return ((v1.__data[lower] & mask) == (v2.__data[lower] & mask));
		} else {
			// the two offsets lay in two different limbs
			// first check the highest limb with the mask
			if ((v1.__data[upper] & umask) != (v2.__data[upper] & umask)) {
				return false;
			}

			// check all limbs in the middle.
			for (int32_t i = upper - 1; i > lower; i--) {
				if (v1.__data[i] != v2.__data[i]) {
					return false;
				}
			}

			// and at the end check the lowest limb.
			if ((v1.__data[lower] & lmask) != (v2.__data[lower] & lmask)) {
				return false;
			}

			return true;
		}
	}


	template<const uint32_t k_lower, const uint32_t k_upper>
	[[nodiscard]] inline constexpr static bool cmp(FqPackedVector const &v1,
	                                               FqPackedVector const &v2) noexcept {
		static_assert(k_upper <= length() && k_lower < k_upper);
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper - 1);
		constexpr T lmask = higher_mask(k_lower);
		constexpr T umask = lower_mask2(k_upper);

		if constexpr (lower == upper) {
			// the two offsets lay in the same limb.
			constexpr T mask = k_upper % limb_bits_width() == 0 ? lmask : (lmask & umask);
			return ((v1.__data[lower] & mask) == (v2.__data[lower] & mask));
		} else {
			// the two offsets lay in two different limbs
			// first check the highest limb with the mask
			if ((v1.__data[upper] & umask) != (v2.__data[upper] & umask)) {
				return false;
			}

			// check all limbs in the middle.
			for (uint32_t i = upper - 1; i > lower; i--) {
				if (v1.__data[i] != v2.__data[i]) {
					return false;
				}
			}

			// and at the end check the lowest limb.
			if ((v1.__data[lower] & lmask) != (v2.__data[lower] & lmask)) {
				return false;
			}

			return true;
		}
	}

	/// checks whether this == obj on the interval [k_lower, ..., k_upper]
	/// the level of the calling 'list' object.
	/// \return
	[[nodiscard]] constexpr inline bool is_equal(const FqPackedVector &obj,
						 						 const uint32_t k_lower=0,
						 						 const uint32_t k_upper=length()) const noexcept {
		return cmp(*this, obj, k_lower, k_upper);
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param obj
	/// \return
	template<const uint32_t k_lower, const uint32_t k_upper>
	[[nodiscard]] constexpr inline bool is_equal(const FqPackedVector &obj) const noexcept {
		return cmp<k_lower, k_upper>(*this, obj);
	}

	/// implements a strict comparison. Call this function if you dont know what
	/// to call. Its the most generic implementaion
	/// and it works for all input.s
	[[nodiscard]] constexpr inline bool is_greater(FqPackedVector const &obj,
						   const uint32_t k_lower = 0,
						   const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		uint32_t lower = round_down_to_limb(k_lower);
		uint32_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T umask = lower_mask2(k_upper);

		if (lower == upper) {
			// the two offsets lay in the same limb.
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & umask);
			return ((__data[lower] & mask) > (obj.__data[lower] & mask));
		} else {
			// the two offsets lay in two different limbs
			ASSERT(lower < upper && lmask != 0 && upper < limbs());
			BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, >, <)
			// check all limbs in the middle
			for (uint64_t i = upper - 1; i > lower; i--) {
				BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], >, <)
			}

			BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, >, <)
			return false;
		}
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param obj
	/// \return
	template<const uint32_t k_lower, const uint32_t k_upper>
	[[nodiscard]] inline bool is_greater(FqPackedVector const &obj) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper - 1);
		constexpr T lmask = higher_mask(k_lower);
		constexpr T umask = lower_mask2(k_upper);
		if constexpr (lower == upper) {// the two offsets lay in the same limb.
			constexpr T mask = k_upper % 64 == 0 ? lmask : (lmask & umask);
			return ((__data[lower] & mask) > (obj.__data[lower] & mask));
		} else {
			// the two offsets lay in two different limbs
			static_assert(lower < upper && lmask != 0 && upper < limbs());

			BINARYCONTAINER_COMPARE_MASKED(__data[upper], obj.__data[upper], umask, >, <)
			// check all limbs in the middle
			for (uint64_t i = upper - 1; i > lower; i--) {
				BINARYCONTAINER_COMPARE(__data[i], obj.__data[i], >, <)
			}

			BINARYCONTAINER_COMPARE_MASKED(__data[lower], obj.__data[lower], lmask, >, <)
			return false;
		}
	}

	/// main comparison function for the < operator. If you dont know what
	/// function to use, use this one. It's the most generic
	/// implementation and works for all inputs.
	inline bool is_lower(FqPackedVector const &obj,
						 const uint32_t k_lower = 0,
						 const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		const uint32_t lower = round_down_to_limb(k_lower);
		const uint32_t upper = round_down_to_limb(k_upper - 1);
		const T lmask = higher_mask(k_lower);
		const T umask = lower_mask2(k_upper);

		if (lower == upper) {// the two offsets lay in the same limb.
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & umask);
			return ((__data[lower] & mask) < (obj.__data[lower] & mask));
		} else {// the two offsets lay in two different limbs
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
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param obj
	/// \return
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_lower(FqPackedVector const &obj) const noexcept {
		static_assert(k_upper <= length() && k_lower < k_upper);
		constexpr uint32_t lower = round_down_to_limb(k_lower);
		constexpr uint32_t upper = round_down_to_limb(k_upper - 1);
		constexpr T lmask = higher_mask(k_lower);
		constexpr T umask = lower_mask2(k_upper);

		if constexpr (lower == upper) {
			// the two offsets lay in the same limb.
			constexpr T mask = k_upper % 64 == 0 ? lmask : (lmask & umask);
			return ((__data[lower] & mask) < (obj.__data[lower] & mask));
		} else {// the two offsets lay in two different limbs
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
	}


	/// neg is an empty operation
	constexpr inline void neg(const uint16_t k_lower=0,
	                          const uint16_t k_upper=length()) noexcept {
		// do nothing.
		(void) k_lower;
		(void) k_upper;
	}

	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline void neg() noexcept {
		// do nothing.
		(void) k_lower;
		(void) k_upper;
	}


	/// full length addition.
	/// this += v
	constexpr inline void add(FqPackedVector const &v) noexcept {
		add(*this, *this, v);
	}

	/// this function does a full length addition
	/// v3 = v1 + v2
	constexpr inline static void add(FqPackedVector &v3,
	                                 FqPackedVector const &v1,
	                                 FqPackedVector const &v2) noexcept {
		add(v3.ptr(), v1.ptr(), v2.ptr());
	}

	/// NOTE: little hack, which makes the interaction between
	/// this class and the binary matrix easier
	constexpr inline static void add(FqPackedVector &v3,
	                                 FqPackedVector const &v1,
									 const T *v2) noexcept {
		add(v3.ptr(), v1.ptr(), v2);
	}
	/// full length addition
	/// v3 = v1 + v2
	constexpr inline static void add(T *v3,
							         T const *v1,
							         T const *v2) noexcept {
		return add(v3, v1, v2, limbs());
	}

	/// probably addition on full length
	/// v3 = v1 ^ v2
	/// \tparam align: if set to `true` the internal simd functions will use
	///                aligned instructions.
	template<const bool align = false>
	constexpr static inline void add(T *v3,
									 T const *v1,
									 T const *v2,
									 const uint32_t limbs) noexcept {
		uint32_t i = 0;
		constexpr uint32_t limb_size = 256u / (sizeof(T)*8);

		LOOP_UNROLL()
		for (; i + limb_size <= limbs; i += limb_size) {
			const uint32x8_t x_ = uint32x8_t::load<align>((uint32_t *)(v1 + i));
			const uint32x8_t y_ = uint32x8_t::load<align>((uint32_t *)(v2 + i));
			const uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store(v3 + i, z_);
		}

		// tail operation
		for (; i < limbs; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}
	}



	/// windowed addition.
	/// this += v [k_lower, k_upper)
	constexpr inline bool add(FqPackedVector const &v,
							  const uint32_t k_lower,
							  const uint32_t k_upper,
	                          const uint32_t norm=-1) noexcept {
		return add(*this, *this, v, k_lower, k_upper, norm);
	}

	/// this function does a full length addition
	/// v3 = v1 + v2
	constexpr inline static bool add(FqPackedVector &v3,
	                                 FqPackedVector const &v1,
	                                 FqPackedVector const &v2,
									 const uint32_t k_lower,
									 const uint32_t k_upper,
									 const uint32_t norm=-1) noexcept {
		return add(v3.ptr(), v1.ptr(), v2.ptr(), k_lower, k_upper, norm);
	}

	/// full length addition
	/// v3 = v1 + v2
	constexpr inline static bool add(T *v3,
									 T const *v1,
									 T const *v2,
							  		 const uint32_t k_lower,
							  		 const uint32_t k_upper,
	                          		 const uint32_t norm) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper);
		if (norm == uint32_t(-1)) {
			const T lmask = higher_mask(k_lower % limb_bits_width());
			const T rmask = lower_mask2(k_upper % limb_bits_width());
			const int64_t lower_limb = k_lower / limb_bits_width();
			const int64_t higher_limb = (k_upper - 1) / limb_bits_width();

			if (lower_limb == higher_limb) {
				const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
				T tmp1 = (v3[lower_limb] & ~(mask));
				T tmp2 = (v1[lower_limb] ^ v2[lower_limb]) & mask;
				v3[lower_limb] = tmp1 ^ tmp2;
				return false;
			}

			// todo avx512 support
			constexpr uint32_t limb_size = 256u/(sizeof(T)*8);
			uint32_t i = lower_limb + 1;
			for (; i + limb_size <= higher_limb; i += limb_size) {
				uint32x8_t x_ = uint32x8_t::load<false>((uint32_t *)(v1 + i));
				uint32x8_t y_ = uint32x8_t::load<false>((uint32_t *)(v2 + i));
				uint32x8_t z_ = x_ ^ y_;
				uint32x8_t::store((uint32_t *)(v3 + i), z_);
			}

			for (; i < higher_limb; ++i) {
				v3[i] = v1[i] ^ v2[i];
			}

			// do the remaining stuff
			T tmp1 = (v1[lower_limb] ^ v2[lower_limb]) & lmask;
			T tmp2 = (v1[higher_limb] ^ v2[higher_limb]) & rmask;
			T tmp11 = (v3[lower_limb] & ~(lmask));
			T tmp21 = (v3[higher_limb] & ~(rmask));

			v3[lower_limb] = tmp1 ^ tmp11;
			v3[higher_limb] = tmp2 ^ tmp21;
			return false;
		} else {
			const uint32_t cnorm = add_weight(v3, v1, v2, k_lower, k_upper);
			return cnorm >= norm;
		}
	}


	// add between the coordinate l, h
	template<const uint32_t k_lower,
	         const uint32_t k_upper,
	         const uint32_t norm=-1u>
	__FORCEINLINE__ static bool add(FqPackedVector &v3,
	                                FqPackedVector const &v1,
	                                FqPackedVector const &v2) noexcept {
		if constexpr (norm != uint32_t(-1)) {
			return add_weight<k_lower, k_upper>(v3, v1, v2);
		}

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
			return false;
		}

		constexpr uint32_t limb_size = 256u/(sizeof(T)*8);
		uint32_t i = lower_limb + 1;
		for (; i + limb_size <= higher_limb; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<false>((uint32_t *)(v1.ptr() + i));
			uint32x8_t y_ = uint32x8_t::load<false>((uint32_t *)(v2.ptr() + i));
			uint32x8_t z_ = x_ ^ y_;
			uint32x8_t::store((uint32_t *)(v3.ptr() + i), z_);
		}

		for (; i < higher_limb; ++i) {
			v3[i] = v1[i] ^ v2[i];
		}

		T tmp1 = (v1.__data[lower_limb] ^ v2.__data[lower_limb]) & lmask;
		T tmp2 = (v1.__data[higher_limb] ^ v2.__data[higher_limb]) & rmask;
		T tmp11 = (v3.__data[lower_limb] & ~(lmask));
		T tmp21 = (v3.__data[higher_limb] & ~(rmask));

		v3.__data[lower_limb] = tmp1 ^ tmp11;
		v3.__data[higher_limb] = tmp2 ^ tmp21;
		return false;
	}


	// calculates the sum v3=v1+v2 and returns the hamming weight of v3
	inline constexpr static uint32_t add_weight(T *v3,
	                                            T const *v1,
	                                            T const *v2) noexcept {
		uint32_t cnorm = 0;
		// TODO optimize with avx
		for (uint32_t i = 0; i < limbs(); ++i) {
			v3[i] = v1[i] ^ v2[i];
			cnorm += popcnt_T(v3[i]);
		}

		return cnorm;
	}

	inline constexpr static uint32_t add_weight(FqPackedVector &v3,
	                                            FqPackedVector const &v1,
	                                            FqPackedVector const &v2) noexcept {
		return add_weight(v3.ptr(), v1.ptr(), v2.ptr());
	}


	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr static uint32_t add_weight(FqPackedVector &v3,
	                                     FqPackedVector const &v1,
	                                     FqPackedVector const &v2) noexcept {
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

	constexpr static uint32_t add_weight(FqPackedVector v3,
	                                     FqPackedVector const &v1,
	                                     FqPackedVector const &v2,
										 const uint32_t k_lower,
										 const uint32_t k_upper) noexcept {
		return add_weight(v3.ptr(), v1.ptr(), v2.ptr(), k_lower, k_upper);
	}

	constexpr inline static uint32_t add_weight(T *v3,
									     T const *v1,
									     T const *v2,
										 const uint32_t k_lower,
	                                     const uint32_t k_upper) noexcept {
		ASSERT(k_upper <= length() && k_lower < k_upper && 0 < k_upper);

		uint32_t cnorm = 0;
		const T lmask = higher_mask(k_lower % limb_bits_width());
		const T rmask = lower_mask2(k_upper % limb_bits_width());
		const int64_t lower_limb = k_lower / limb_bits_width();
		const int64_t higher_limb = (k_upper - 1) / limb_bits_width();

		if (lower_limb == higher_limb) {
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			T tmp1 = (v3[lower_limb] & ~(mask));
			T tmp2 = (v1[lower_limb] ^ v2[lower_limb]) & mask;
			v3[lower_limb] = tmp1 ^ tmp2;
			auto b = popcnt_T(tmp2);
			return b;
		}

		LOOP_UNROLL();
		for (int64_t i = lower_limb + 1; i < higher_limb; ++i) {
			v3[i] = v1[i] ^ v2[i];
			cnorm += popcnt_T(v3[i]);
		}

		T tmp1 = (v1[lower_limb] ^ v2[lower_limb]) & lmask;
		T tmp2 = (v1[higher_limb] ^ v2[higher_limb]) & rmask;
		T tmp11 = (v3[lower_limb] & ~(lmask));
		T tmp21 = (v3[higher_limb] & ~(rmask));

		v3[lower_limb] = tmp1 ^ tmp11;
		v3[higher_limb] = tmp2 ^ tmp21;

		cnorm += popcnt_T(tmp1);
		cnorm += popcnt_T(tmp2);

		return cnorm;
	}







	// full length subtraction=addition in F_2
	inline int sub(FqPackedVector const &v) noexcept {
		return this->add(v);
	}

	/// alias for add
	inline void sub(FqPackedVector const &v,
	                const uint32_t k_lower,
	                const uint32_t k_upper) noexcept {
		return add(v, k_lower, k_upper);
	}

	/// alias for add
	inline constexpr static int sub(FqPackedVector &v3,
	                                FqPackedVector const &v1,
	                                FqPackedVector const &v2) noexcept {
		add(v3, v1, v2);
		return 0; // always return it's ok and doesn't need to be filtered
	}

	inline constexpr static bool sub(FqPackedVector &v3,
	                                 FqPackedVector const &v1,
									T const *v2) noexcept {
		add(v3.ptr(), v1.ptr(), v2);
		return false; // always return it's ok and doesn't need to be filtered
	}

	/// alias for add
	inline constexpr static bool sub(FqPackedVector &v3,
	                                 FqPackedVector const &v1,
	                                 FqPackedVector const &v2,
	                                 const uint32_t k_lower,
	                                 const uint32_t k_upper) noexcept {
		add(v3, v1, v2, k_lower, k_upper);
		return false;
	}

	/// alias for add
	inline static bool sub(FqPackedVector &v3,
	                       FqPackedVector const &v1,
	                       FqPackedVector const &v2,
	                       const uint32_t k_lower,
	                       const uint32_t k_upper,
	                       const uint32_t norm) noexcept {
		return add(v3, v1, v2, k_lower, k_upper, norm);
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \tparam norm
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	template<const uint32_t k_lower,
	         const uint32_t k_upper,
	         const uint32_t norm=-1u>
	__FORCEINLINE__ static bool sub(FqPackedVector &v3,
	                                FqPackedVector const &v1,
	                                FqPackedVector const &v2) noexcept {
		return add<k_lower, k_upper, norm>(v3, v1, v2);
	}

	///
	/// \param v3
	/// \param v1
	/// \param v2
	/// \return
	constexpr inline static void sub(T *v3,
									 T const *v1,
									 T const *v2) noexcept {
		return add(v3, v1, v2, limbs());
	}

	///
	/// \param v3
	/// \param v1
	/// \param v2
	/// \param k_lower
	/// \param k_upper
	/// \param norm
	/// \return
	constexpr inline static bool mul(FqPackedVector &v3,
	                                 FqPackedVector const &v1,
	                                 FqPackedVector const &v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=length(),
									 const uint32_t norm=-1) noexcept {
		(void)norm;
		const T lmask = higher_mask(k_lower % limb_bits_width());
		const T rmask = lower_mask2(k_upper % limb_bits_width());
		const int64_t lower_limb = k_lower / limb_bits_width();
		const int64_t higher_limb = (k_upper - 1) / limb_bits_width();

		if (lower_limb == higher_limb) {
			const T mask = k_upper % 64 == 0 ? lmask : (lmask & rmask);
			T tmp1 = (v3[lower_limb] & ~(mask));
			T tmp2 = (v1[lower_limb] ^ v2[lower_limb]) & mask;
			v3[lower_limb] = tmp1 & tmp2;
			return false;
		}

		// todo avx512 support
		constexpr uint32_t limb_size = 256u/(sizeof(T)*8);
		uint32_t i = lower_limb + 1;
		for (; i + limb_size <= higher_limb; i += limb_size) {
			uint32x8_t x_ = uint32x8_t::load<false>((uint32_t *)(v1 + i));
			uint32x8_t y_ = uint32x8_t::load<false>((uint32_t *)(v2 + i));
			uint32x8_t z_ = x_ & y_;
			uint32x8_t::store((uint32_t *)(v3 + i), z_);
		}

		for (; i < higher_limb; ++i) {
			v3[i] = v1[i] & v2[i];
		}

		// do the remaining stuff
		T tmp1 = (v1[lower_limb] ^ v2[lower_limb]) & lmask;
		T tmp2 = (v1[higher_limb] ^ v2[higher_limb]) & rmask;
		T tmp11 = (v3[lower_limb] & ~(lmask));
		T tmp21 = (v3[higher_limb] & ~(rmask));

		v3[lower_limb] = tmp1 & tmp11;
		v3[higher_limb] = tmp2 & tmp21;
		return false;
	}

	///
	/// \param v3
	/// \param v1
	/// \param v2
	/// \param k_lower
	/// \param k_upper
	/// \param norm
	/// \return
	constexpr inline static bool scalar(FqPackedVector &v3,
	                                    FqPackedVector const &v1,
									 DataType const v2,
									 const uint32_t k_lower=0,
									 const uint32_t k_upper=length(),
									 const uint32_t norm=-1) noexcept {
		(void)norm;
		if (v2) {
			FqPackedVector::set(v3, v1, k_lower, k_upper);
			return false;
		} else {
			v3.zero(k_lower, k_upper);
		}
	}



	/// IMPORTANT: k_lower < k_upper is enforced.
	/// sets v1 = v2[k_lower, ..., k_upper].
	/// Does not change anything else.
	inline constexpr static void set(FqPackedVector &v1, FqPackedVector const &v2,
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
	constexpr static inline void sll(FqPackedVector &out,
	                               const FqPackedVector &in,
	                               const uint32_t s) noexcept {
		out.zero();

		ASSERT(s < length());
		for (uint32_t j = 0; j < length() - s; ++j) {
			const auto bit = in.get_bit_shifted(j);
			out.write_bit(j + s, bit);
		}
	}

	///  out[0: ] = in[s:]
	constexpr static inline void slr(FqPackedVector &out,
						   const FqPackedVector &in,
						   const uint32_t s) noexcept {
		ASSERT(s < length());
		for (uint32_t j = s; j < length(); ++j) {
			out.write_bit(j-s, in.get_bit_shifted(j));
		}
		for (uint32_t j = s; j < length(); ++j) {
			out.write_bit(j, 0);
		}
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
	constexpr static inline uint32_t dist(const FqPackedVector &a,
	                                      const FqPackedVector &b) noexcept {
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


	// hack it like its C++
	class reference {
		friend class BinaryContainer;

		// pointer to the limb
		T *wp;

		// bit position in the whole data const_array.
		const size_t mask_pos;

		// left undefined
		reference();

	public:
		constexpr reference(const FqPackedVector &b, const size_t pos) : mask_pos(mask(pos)) {
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
		constexpr reference &operator=(const reference &j) noexcept {
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

	/// wrapper around `print`
	void print_binary(const uint32_t k_lower = 0,
	                  const uint32_t k_upper = length()) const noexcept {
		print(k_lower, k_upper);
	}

	/// print some information
	/// \param k_lower lower limit to print (included)
	/// \param k_upper higher limit to print (not included)
	void print(const uint32_t k_lower = 0,
	           const uint32_t k_upper = length()) const noexcept {
		ASSERT(k_lower < length() && k_upper <= length() && k_lower < k_upper);
		for (uint64_t i = k_lower; i < k_upper; ++i) {
			std::cout << data(i) << "";
		}
		std::cout << "\n";
	}

	//T data(uint64_t index) { ASSERT(index < length); return get_bit_shifted(index); }
	[[nodiscard]] bool data(const uint64_t index) const noexcept {
		ASSERT(index < length());
		return get_bit_shifted(index);
	}

	// simple hash function
	template<const uint32_t l, const uint32_t h>
	[[nodiscard]] constexpr inline size_t hash() const noexcept {
		static_assert(l < h);
		static_assert(h <= length());
		static_assert((h-l) <= 64, "Sorry, but hashing down to more than 64 bits is not possible");

		constexpr uint32_t qbits = 1;
		constexpr uint32_t bits = limb_bits_width();
		constexpr uint32_t lq = l*qbits;
		constexpr uint32_t hq = h*qbits;
		constexpr uint32_t llimb = lq / bits;
		constexpr uint32_t hlimb  = hq%bits == 0u ? llimb : hq / bits;
		constexpr uint32_t lprime = lq % bits;
		constexpr uint32_t hprime = (hq%bits) == 0 ? bits : hq % bits;

		// easy case: lower limit and upper limit
		// are in the same limb
		if constexpr (llimb == hlimb) {
			static_assert(lprime < hprime);
			static_assert((hprime - lprime) <= (sizeof(T) * 8u));

			constexpr T diff1 = hprime - lprime;
			static_assert (diff1 <= bits);
			constexpr T diff2 = bits - diff1;
			constexpr T mask = -1ull >> diff2;
			const T b = __data[llimb] >> lprime;
			const T c = b & mask;
			return c;
		}

		static_assert(llimb <= hlimb);
		static_assert((hlimb - llimb) <= 1u); // note could be extended

		constexpr T lmask = T(-1ull) << lprime;
		constexpr T hmask = T(-1ull) >> ((bits - hprime) % bits);

		// not so easy case: lower limit and upper limit are
		// on seperate limbs
		T data = (__data[llimb] & lmask) >> lprime;
		data  ^= (__data[hlimb] & hmask) << ((bits - lprime) % bits);
		return data;
	}
	[[nodiscard]] constexpr inline size_t hash(const uint32_t l,
	                                           const uint32_t h) const noexcept {
		ASSERT(l < h);
		ASSERT(h <= length());

		const uint32_t bits = limb_bits_width();
		const uint32_t llimb = l / bits;
		const uint32_t hlimb  = h%bits == 0u ? llimb : h / bits;
		const uint32_t lprime = l % bits;
		const uint32_t hprime = (h%bits) == 0 ? bits : h % bits;

		// easy case: lower limit and upper limit
		// are in the same limb
		if (llimb == hlimb) {
			ASSERT(lprime < hprime);
			ASSERT((hprime - lprime) <= (sizeof(T) * 8u));

			const T diff1 = hprime - lprime;
			ASSERT (diff1 <= bits);
			const T diff2 = bits - diff1;
			const T mask = -1ull >> diff2;
			const T b = __data[llimb] >> lprime;
			const T c = b & mask;
			return c;
		}

		ASSERT(llimb <= hlimb);
		ASSERT((hlimb - llimb) <= 1u); // note could be extended

		const T lmask = T(-1ull) << lprime;
		const T hmask = T(-1ull) >> ((bits - hprime) % bits);

		// not so easy case: lower limit and upper limit are
		// on seperate limbs
		T data = (__data[llimb] & lmask) >> lprime;
		data  ^= (__data[hlimb] & hmask) << ((bits - lprime) % bits);
		return data;
	}
	// full length hasher
	[[nodiscard]] constexpr inline auto hash() const noexcept {
		return *this;
		// using S = TxN_t<T, limbs()>;
		// const S *s = (S *)__data.data();
		// const auto t = Hash<S> (s);
		// return t;
	}

	///
	template<const uint32_t l, const uint32_t h>
	constexpr static inline bool is_hashable() noexcept {
		if constexpr (h == l) { return false; }
		constexpr size_t t1 = h-l;
		return t1 <= 64u;
	}
	constexpr static bool is_hashable(const uint32_t l,
							   		  const uint32_t h) noexcept {
		ASSERT(h > l);
		const size_t t1 = h-l;
		return t1 <= 64u;
	}



/// TODO make this an field in the config
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
	[[nodiscard]] __FORCEINLINE__ constexpr static bool binary() noexcept { return true; }
	[[nodiscard]] __FORCEINLINE__ constexpr static uint32_t size() noexcept { return length(); }
	[[nodiscard]] __FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return (length() + limb_bits_width() - 1) / limb_bits_width(); }
	/// returns size of a single element in this container in bits
	[[nodiscard]] static constexpr inline size_t sub_container_size() noexcept {
		return 1;
	}
	[[nodiscard]] __FORCEINLINE__ constexpr static uint32_t bytes() noexcept {
#ifdef BINARY_CONTAINER_ALIGNMENT
		return alignment() / 8;
#else
		return limbs() * sizeof(T);
#endif
	}

	[[nodiscard]] __FORCEINLINE__ T *ptr() noexcept { return __data.data(); };
	[[nodiscard]] const __FORCEINLINE__ T *ptr() const noexcept { return __data.data(); };
	[[nodiscard]] __FORCEINLINE__ T ptr(const size_t i) noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};
	[[nodiscard]] const __FORCEINLINE__ T ptr(const size_t i) const noexcept {
		ASSERT(i < limbs());
		return __data[i];
	};

	// get the internal data vector
	[[nodiscard]] __FORCEINLINE__ std::array<T, compute_limbs()> &data() noexcept { return __data; };
	[[nodiscard]] __FORCEINLINE__ const std::array<T, compute_limbs()> &data() const noexcept { return __data; };

	// returns `true` as this class implements an optimized arithmetic, and not a generic one.
	[[nodiscard]] __FORCEINLINE__ static constexpr bool optimized() noexcept { return true; };

	///
	constexpr static void info() noexcept {
		std::cout << "{ name: \"kAryContainerMeta\""
				  << ", n: " << n
				  << ", q: " << q
				  << ", sizeof(T): " << sizeof(T)
				  << "}\n";
	}
private:
	// actual data container.
	std::array<T, compute_limbs()> __data;
};

///
template<const uint32_t n>
using BinaryVector = FqPackedVector<n, 2, uint64_t>;

template<const uint64_t n, typename T>
constexpr inline bool operator==(const FqPackedVector<n, 2, T> &a,
                                 const FqPackedVector<n, 2, T> &b) noexcept {
	return a.is_equal(b);
}
template<const uint64_t n, typename T>
constexpr inline bool operator<(const FqPackedVector<n, 2, T> &a,
                                const FqPackedVector<n, 2, T> &b) noexcept {
	return a.is_lower(b);
}
template<const uint64_t n, typename T>
constexpr inline bool operator>(const FqPackedVector<n, 2, T> &a,
                                const FqPackedVector<n, 2, T> &b) noexcept {
	return a.is_greater(b);
}

template<uint64_t _n,
        typename T=uint64_t>
std::ostream &operator<<(std::ostream &out,
                         const FqPackedVector<_n, 2, T> &obj) {
	constexpr bool print_weight = true;
	for (size_t i = 0; i < obj.length(); ++i) {
		out << obj[i];
	}

	if constexpr (print_weight) {
		std::cout << ", (wt=" << std::dec << obj.popcnt() << ")";
	}

	return out;
}

#endif//SMALLSECRETLWE_CONTAINER_H
