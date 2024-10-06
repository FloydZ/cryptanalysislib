#ifndef CRYPTANALYSISLIB_RANDOM_H
#define CRYPTANALYSISLIB_RANDOM_H

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "algorithm/rotate.h"

// floor( ( (1+sqrt(5))/2 ) * 2**64 MOD 2**64)
#define GOLDEN_GAMMA UINT64_C(0x9E3779B97F4A7C15)

namespace cryptanalysislib {
	/// TODO write c++ random function, like class abstraction
namespace random::internal {

/// super rng values
static uint64_t random_x = 123456789u, random_y = 362436069u, random_z = 521288629u;

/// NOTE: this function cannot fail
/// \param i see
/// \return true on success
[[nodiscard]] constexpr static inline bool xorshf96_seed(const uint64_t seed) noexcept {
	random_x += seed;
	random_y = random_x * 4095834;
	random_z = random_x + random_y * 98798234;
	return true;
}

[[nodiscard]] constexpr static inline bool xorshf96_seed() noexcept {
	uint64_t new_s[3];
	FILE *urandom_fp;

	urandom_fp = fopen("/dev/urandom", "r");
	if (urandom_fp == nullptr) return 0;
	if (fread(&new_s, 8, 3, urandom_fp) != 2) {
		return false;
	}
	fclose(urandom_fp);

	random_y += new_s[1];
	random_z += new_s[2];
	return xorshf96_seed(new_s[0]);
}

/// period 2^96-1
/// \return
[[nodiscard]] static inline uint64_t xorshf96() noexcept {
	random_x ^= random_x << 16u;
	random_x ^= random_x >> 5u;
	random_x ^= random_x << 1u;

	const uint64_t t = random_x;
	random_x = random_y;
	random_y = random_z;
	random_z = t ^ random_x ^ random_y;

	return random_z;
}

/// n = size of buffer in bytes
/// return 0 on success (cannot fail)
static inline void xorshf96_random_data(uint8_t *buf,
		                           		const size_t n) noexcept {
	auto *a = (uint64_t *) buf;

	const uint32_t rest = n % 8;
	const size_t limit = n / 8;
	size_t i = 0;

	for (; i < limit; ++i) {
		a[i] = xorshf96();
	}

	// last limb
	auto *b = (uint8_t *) buf;
	b += n - rest;
	uint64_t limb = xorshf96();
	for (size_t j = 0; j < rest; ++j) {
		b[j] = (limb >> (j * 8u)) & 0xFFu;
	}
}

///
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
[[nodiscard]] static inline T xorshf96_random_data() noexcept {
	return xorshf96();
}

/// This is xoroshiro128+ 1.0, our best and fastest small-state generator
/// for floating-point numbers. We suggest to use its upper bits for
/// floating-point generation, as it is slightly faster than xoroshiro128**. It
/// passes all tests we are aware of except for the four lower bits, which might
/// fail linearity tests (and just those), so if low linear complexity is not
/// considered an issue (as it is usually the case) it can be used to generate
/// 64-bit outputs, too; moreover, this generator has a very mild Hamming-weight
/// dependency making our test (http://prng.di.unimi.it/hwd.php) fail after 5 TB
/// of output; we believe this slight bias cannot affect any application. If you
/// are concerned, use xoroshiro128++, xoroshiro128** or xoshiro256+.
///
/// We suggest to use a sign test to extract a random Boolean value, and right
/// shifts to extract subsets of bits.
/// The state must be seeded so that it is not everywhere zero. If you have a
/// 64-bit seed, we suggest to seed a splitmix64 generator and use its output to
/// fill s.
///
/// NOTE: the parameters (a=24, b=16, b=37) of this version give slightly better
/// results in our test than the 2016 version (a=55, b=14, c=36).


/// "randomly" choosen start values to the xorshf128 prng
static uint64_t __xorshf128_S0 = 2837468099234763274;
static uint64_t __xorshf128_S1 = 998234767632513414;

/// \return random uint64_t
[[nodiscard]] static inline uint64_t xorshf128_random_data() noexcept {
	const uint64_t s0 = __xorshf128_S0;
	uint64_t s1 = __xorshf128_S1;
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	__xorshf128_S0 = rotl(s0, 24) ^ s1 ^ (s1 << 16);
	__xorshf128_S1 = rotl(s1, 37);

	return result;
}

/// This is the jump function for the generator. It is equivalent
/// to 2^64 calls to next(); it can be used to generate 2^64 non-overlapping
/// subsequences for parallel computations.
constexpr static inline void jump() noexcept {
	static const uint64_t JUMP[] = {0xdf900294d8f554a5, 0x170865df4b3201fc};

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for (unsigned i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
		for (int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= __xorshf128_S0;
				s1 ^= __xorshf128_S1;
			}
			const uint64_t t = xorshf128_random_data();
			(void)t;
		}

	__xorshf128_S0 = s0;
	__xorshf128_S1 = s1;
}

/// This is the long-jump function for the generator. It is equivalent to
/// 2^96 calls to next(); it can be used to generate 2^32 starting points, from
/// each of which jump() will generate 2^32 non-overlapping subsequences for
/// parallel distributed computations.
constexpr static inline void long_jump() noexcept {
	static const uint64_t LONG_JUMP[] = {0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1};

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for (unsigned i = 0; i < sizeof(LONG_JUMP) / sizeof(*LONG_JUMP); i++)
		for (int b = 0; b < 64; b++) {
			if (LONG_JUMP[i] & UINT64_C(1) << b) {
				s0 ^= __xorshf128_S0;
				s1 ^= __xorshf128_S1;
			}
			const uint64_t t = xorshf128_random_data();
			(void)t;
		}

	__xorshf128_S0 = s0;
	__xorshf128_S1 = s1;
}

static inline bool xorshf128_seed() noexcept {
	uint64_t new_s[2];
	FILE *urandom_fp;

	urandom_fp = fopen("/dev/urandom", "r");
	if (urandom_fp == NULL) return 0;
	if (fread(&new_s, 8, 2, urandom_fp) != 2) {
		return false;
	}
	fclose(urandom_fp);

	__xorshf128_S0 = new_s[0];
	__xorshf128_S1 = new_s[1];

	return true;
}



/// pcg64 
/// SCR: https://github.com/lemire/batched_random/blob/main/src/pcg64.h
///     based on original code by M. O'Neill
///     modified by Floyd to match a C++ env.

/// TODO move this macro into uint128_t
#define PCG_128BIT_CONSTANT(high, low) ((((__uint128_t)high) << 64) + low)
#define PCG_DEFAULT_MULTIPLIER_128                                             \
  PCG_128BIT_CONSTANT(2549297995355413924ULL, 4865540595714422341ULL)
#define PCG_DEFAULT_INCREMENT_128                                              \
  PCG_128BIT_CONSTANT(6364136223846793005ULL, 1442695040888963407ULL)


static __uint128_t pcg_state_setseq_128_state;
static __uint128_t pcg_state_setseq_128_inc;

constexpr static inline void pcg_setseq_128_step_r() noexcept {
    pcg_state_setseq_128_state = pcg_state_setseq_128_state*PCG_DEFAULT_MULTIPLIER_128 
                                + pcg_state_setseq_128_inc;
}

inline void pcg_setseq_128_srandom_r(__uint128_t initstate,
                                     __uint128_t initseq) {
  pcg_state_setseq_128_state = 0U;
  pcg_state_setseq_128_inc = (initseq << 1u) | 1u;
  pcg_setseq_128_step_r();
  pcg_state_setseq_128_state += initstate;
  pcg_setseq_128_step_r();
}

///
[[nodiscard]] constexpr static inline uint64_t pcg_output_xsl_rr_128_64() noexcept {
  return rotr(((uint64_t)(pcg_state_setseq_128_state >> 64u)) ^ (uint64_t)pcg_state_setseq_128_state, 
              (unsigned int)(pcg_state_setseq_128_state >> 122u));
}

/// TODO benchmark against all other generators
[[nodiscard]] constexpr static inline uint64_t pcg64_random_data() noexcept {
  pcg_setseq_128_step_r();
  return pcg_output_xsl_rr_128_64();
}


/// splitmix64 
[[nodiscard]] constexpr static inline uint64_t splitmix64_stateless(const uint64_t index) noexcept {
    uint64_t z = (index * UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

[[nodiscard]] constexpr static inline uint64_t splitmix64_r(uint64_t *seed) noexcept {
    uint64_t z = (*seed += GOLDEN_GAMMA);
    // David Stafford's Mix13 for MurmurHash3's 64-bit finalizer
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

// returns the value of splitmix64 "offset" steps from seed
[[nodiscard]] constexpr static inline uint64_t splitmix64_stateless_offset(uint64_t seed,
                                                                           const uint64_t offset) noexcept {
    seed += offset * GOLDEN_GAMMA;
    return splitmix64_r(&seed);
}

static __uint128_t g_lehmer64_state = UINT64_C(0x853c49e6748fea9b);

///
/// D. H. Lehmer, Mathematical methods in large-scale computing units.
/// Proceedings of a Second Symposium on Large Scale Digital Calculating
/// Machinery;
/// Annals of the Computation Laboratory, Harvard Univ. 26 (1951), pp. 141-146.
///
constexpr static inline void lehmer64_seed(const uint64_t seed) noexcept {
    g_lehmer64_state = (((__uint128_t)splitmix64_stateless(seed)) << 64) +
                                      splitmix64_stateless(seed + 1);
}

[[nodiscard]] constexpr static inline uint64_t lehmer64_random_data() noexcept {
    g_lehmer64_state *= UINT64_C(0xda942042e4dd58b5);
    return (uint64_t)(g_lehmer64_state >> 64);
}

} // namespace random::internal


///
/// \param buf out
/// \param n size in bytes
/// \return 0 on success, 1 on error
static inline void rng(uint8_t *buf,
	                   const size_t n) noexcept {
	random::internal::xorshf96_random_data(buf, n);
}

// TODO second rng_seed function for /dev/urandom
/// seed the rng instance
/// \param seed
constexpr static inline void rng_seed(const uint64_t seed) noexcept {
	const bool t = random::internal::xorshf96_seed(seed);
	(void)t;
}

/// \return a uniform rng `uint64_t`
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T rng() noexcept {
	return random::internal::xorshf96_random_data<T>();
}

/// \return a uniform (not really) uint64 % limit
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline uint64_t rng(const T limit) noexcept {
	ASSERT(limit > 0);
	return random::internal::xorshf96_random_data<T>() % limit;
}

/// \return a rng element from [l, h)
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T rng(const T l,
                                  const T h) noexcept {
	return l + rng<T>(h - l);
}

/// \param w hamming weight of the output element
/// \return a weight w type `T` element
template<typename T>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
[[nodiscard]] constexpr static T rng_weighted(const uint32_t w) noexcept {
	assert(w < (sizeof(T) * 8));

	T ret = (1u << w) - 1u;
	for (uint32_t i = 0; i < w; ++i) {
		const size_t to_pos = rng() % ((sizeof(T) * 8) - i);
		const size_t from_pos = i;

		const T from_mask = 1u << from_pos;
		const T to_mask = 1u << to_pos;

		const T from_read = (ret & from_mask) >> from_pos;
		const T to_read = (ret & to_mask) >> to_pos;

		ret ^= (-from_read ^ ret) & (1ul << to_pos);
		ret ^= (-to_read ^ ret) & (1ul << from_pos);
	}

	return ret;
}

/// Original Source
/// https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2024/08/16/include/batched_shuffle.h
/// Nevin Brackett-Rozinsky, Daniel Lemire, Batched Ranged Random Integer Generation,
/// Software: Practice and Experience (to appear) 
/// Daniel Lemire, Fast Random Integer Generation in an Interval, ACM 
/// Transactions on Modeling and Computer Simulation, Volume 29 Issue 1, February 2019 
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline T rng_v2(const uint64_t range) noexcept {
	__uint128_t random64bit, multiresult;
	uint64_t leftover;
	uint64_t threshold;
	random64bit = rng<uint64_t>();
	multiresult = random64bit * range;
	leftover = (uint64_t)multiresult;
	if (leftover < range) {
		threshold = -range % range;
		while (leftover < threshold) {
			random64bit = rng<uint64_t>();
			multiresult = random64bit * range;
			leftover = (uint64_t)multiresult;
		}
	}
	return (T)(multiresult >> 64); // [0, range)
}

// TODO?
// product_bound can be any integer >= range1*range2
// it may be updated to become range1*range2
template <class URBG>
std::pair<uint64_t, uint64_t> random_bounded_2(uint64_t range1, uint64_t range2,
                                               uint64_t &product_bound,
                                               URBG &&rng) {
  __uint128_t random64bit, multiresult;
  uint64_t leftover;
  uint64_t threshold;
  random64bit = rng();
  multiresult = random64bit * range1;
  leftover = (uint64_t)multiresult;
  uint64_t result1 = (uint64_t)(multiresult >> 64); // [0, range1)
  multiresult = leftover * range2;
  leftover = (uint64_t)multiresult;
  uint64_t result2 = (uint64_t)(multiresult >> 64); // [0, range2)
  if (leftover < product_bound) {
    product_bound = range2 * range1;
    if (leftover < product_bound) {
      threshold = -product_bound % product_bound;
      while (leftover < threshold) {
        random64bit = rng();
        multiresult = random64bit * range1;
        leftover = (uint64_t)multiresult;
        result1 = (uint64_t)(multiresult >> 64); // [0, range1)
        multiresult = leftover * range2;
        leftover = (uint64_t)multiresult;
        result2 = (uint64_t)(multiresult >> 64); // [0, range2)
      }
    }
  }
  return std::make_pair(result1, result2);
}

// This is a template function that shuffles the elements in the range [first,
// last).
//
// It is similar to std::shuffle, but it uses a different algorithm.
template <class RandomIt, class URBG>
extern void shuffle_2(RandomIt first, RandomIt last, URBG &&g) {
  uint64_t i = std::distance(first, last);
  for (; i > 1 << 30; i--) {
    uint64_t index = random_bounded(i, g); // index is in [0, i-1]
    std::iter_swap(first + i - 1, first + index);
  }

  // Batches of 2 for sizes up to 2^30 elements
  uint64_t product_bound = i * (i - 1);
  for (; i > 1; i -= 2) {
    auto [index1, index2] = random_bounded_2(i, i - 1, product_bound, g);
    // index1 is in [0, i-1]
    // index2 is in [0, i-2]
    std::iter_swap(first + i - 1, first + index1);
    std::iter_swap(first + i - 2, first + index2);
  }
}

}// namespace cryptanalysislib
#endif//SMALLSECRETLWE_RANDOM_H
