#ifndef CRYPTANALYSISLIB_RANDOM_H
#define CRYPTANALYSISLIB_RANDOM_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

/// super random values
static uint64_t random_x = 123456789u, random_y = 362436069u, random_z = 521288629u;

// Sowas von nicht sicher, Aber egal.
static inline void xorshf96_random_seed(const uint64_t i) noexcept {
	random_x += i;
	random_y = random_x * 4095834;
	random_z = random_x + random_y * 98798234;
}

[[nodiscard]] static inline uint64_t xorshf96() noexcept {//period 2^96-1
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
static inline int xorshf96_fastrandombytes(void *buf, const size_t n) noexcept {
	uint64_t *a = (uint64_t *) buf;

	const uint32_t rest = n % 8;
	const size_t limit = n / 8;
	size_t i = 0;

	for (; i < limit; ++i) {
		a[i] = xorshf96();
	}

	// last limb
	uint8_t *b = (uint8_t *) buf;
	b += n - rest;
	uint64_t limb = xorshf96();
	for (size_t j = 0; j < rest; ++j) {
		b[j] = (limb >> (j * 8u)) & 0xFFu;
	}

	return 0;
}

/// n = bytes
/// returns 0 on success
[[nodiscard]] inline static int xorshf96_fastrandombytes_uint64_array(uint64_t *buf, const size_t n) noexcept {
	xorshf96_fastrandombytes(buf, n);
	return 0;
}

///
[[nodiscard]] static inline uint64_t xorshf96_fastrandombytes_uint64() noexcept {
	return xorshf96();
}

///
/// \param buf out
/// \param n size in bytes
/// \return 0 on success, 1 on error
[[nodiscard]] static inline int fastrandombytes(void *buf, const size_t n) noexcept {
	return xorshf96_fastrandombytes(buf, n);
}

/// seed the random instance
/// \param seed
static inline void random_seed(uint64_t seed) noexcept {
	xorshf96_random_seed(seed);
}

/// \return a uniform random `uint64_t`
[[nodiscard]] static inline uint64_t fastrandombytes_uint64() noexcept {
	return xorshf96_fastrandombytes_uint64();
}

/// \return a uniform (not really) uint64 % limit
[[nodiscard]] static inline uint64_t fastrandombytes_uint64(const uint64_t limit) noexcept {
	return xorshf96_fastrandombytes_uint64() % limit;
}

/// \return a random element from [l, h)
[[nodiscard]] static inline uint64_t fastrandombytes_uint64(const uint64_t l,
                                                            const uint64_t h) noexcept {
	return l + fastrandombytes_uint64(h - l);
}

/// simple C++ wrapper.
/// \tparam T
/// \return a type T uniform random element
template<typename T>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T fastrandombytes_T() noexcept {
	return xorshf96_fastrandombytes_uint64();
}

/// simple C++ wrapper.
/// \tparam T
/// \param limit
/// \return a type T uniform random element % limit
template<typename T>
#if __cplusplus > 201709L
requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T fastrandombytes_T(const T limit) noexcept {
	return xorshf96_fastrandombytes_uint64() % limit;
}

/// simple C++ wrapper.
/// \tparam T
/// \param limit
/// \return a type T uniform random element % limit
template<typename T, const T mod>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T fastrandombytes_T() noexcept {
	// thats an implementation limit
	static_assert(sizeof(T) < 8);
	return fastmod<mod>(fastrandombytes_T<T>);
}

/// simple C++ wrapper.
/// \tparam T base type
/// \param l inclusive lower limit
/// \param h exclusive upper limit
/// \return a type T uniform random element in [l, h)
template<typename T>
#if __cplusplus > 201709L
requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T fastrandombytes_T(const T l, const T h) noexcept {
	return l + fastrandombytes_T<T>(h - l);
}

/// simple C++ wrapper.
/// \tparam T base type
/// \param l inclusive lower limit
/// \param h exclusive upper limit
/// \return a type T uniform random element in [l, h)
template<typename T, const T l, const T h>
#if __cplusplus > 201709L
requires std::is_integral_v<T>
#endif
[[nodiscard]] static inline T fastrandombytes_T() noexcept {
	// thats an implementation limit
	static_assert(sizeof(T) < 8);
	return l + fastmod<h - l>(fastrandombytes_T<T>);
}

/// \param w hamming weight of the output element
/// \return a weight w type `T` element
template<typename T>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
[[nodiscard]] constexpr static T fastrandombytes_weighted(const uint32_t w) noexcept {
	assert(w < (sizeof(T) * 8));

	T ret = (1u << w) - 1u;
	for (uint32_t i = 0; i < w; ++i) {
		const size_t to_pos = fastrandombytes_uint64() % ((sizeof(T) * 8) - i);
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


#include <utility>

/// Original Source
/// https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2024/08/16/include/batched_shuffle.h
///
namespace batched_random {
/** 
 * Nevin Brackett-Rozinsky, Daniel Lemire, Batched Ranged Random Integer Generation, Software: Practice and Experience (to appear) 
 * Daniel Lemire, Fast Random Integer Generation in an Interval, ACM Transactions on Modeling and Computer Simulation, Volume 29 Issue 1, February 2019 
 */
template <class URBG> uint64_t random_bounded(uint64_t range, URBG &&rng) {
  __uint128_t random64bit, multiresult;
  uint64_t leftover;
  uint64_t threshold;
  random64bit = rng();
  multiresult = random64bit * range;
  leftover = (uint64_t)multiresult;
  if (leftover < range) {
    threshold = -range % range;
    while (leftover < threshold) {
      random64bit = rng();
      multiresult = random64bit * range;
      leftover = (uint64_t)multiresult;
    }
  }
  return (uint64_t)(multiresult >> 64); // [0, range)
}




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

} // namespace batched_random
#endif//SMALLSECRETLWE_RANDOM_H
