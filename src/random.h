#ifndef SMALLSECRETLWE_RANDOM_H
#define SMALLSECRETLWE_RANDOM_H

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

static inline uint64_t xorshf96() noexcept {//period 2^96-1
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
/// reutrn 0 on success
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
inline static int xorshf96_fastrandombytes_uint64_array(uint64_t *buf, const size_t n) noexcept {
	xorshf96_fastrandombytes(buf, n);
	return 0;
}

///
inline uint64_t xorshf96_fastrandombytes_uint64() noexcept {
	return xorshf96();
}

///
/// \param buf out
/// \param n size in bytes
/// \return 0 on success, 1 on error
static inline int fastrandombytes(void *buf, const size_t n) noexcept {
	return xorshf96_fastrandombytes(buf, n);
}

/// seed the random instance
/// \param seed
static inline void random_seed(uint64_t seed) noexcept {
	xorshf96_random_seed(seed);
}

/// \return a uniform random `uint64_t`
static inline uint64_t fastrandombytes_uint64() noexcept {
	return xorshf96_fastrandombytes_uint64();
}

/// simple C++ wrapper.
/// \tparam T
/// \return a type T uniform random element
template<typename T>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
static inline T fastrandombytes_T() noexcept {
	return xorshf96_fastrandombytes_uint64();
}

/// creates a weight w uint64_t
/// \param w
/// \return
template<typename T>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
static T fastrandombytes_weighted(const uint32_t w) noexcept {
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
#endif//SMALLSECRETLWE_RANDOM_H
