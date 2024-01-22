#ifndef SMALLSECRETLWE_RANDOM_H
#define SMALLSECRETLWE_RANDOM_H

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <type_traits>

/// super random values
static uint64_t random_x=123456789u, random_y=362436069u, random_z=521288629u;

// Sowas von nicht sicher, Aber egal.
static inline void xorshf96_random_seed(const uint64_t i) noexcept {
	random_x += i;
	random_y = random_x*4095834;
	random_z = random_x + random_y*98798234;
}

static inline uint64_t xorshf96() noexcept {          //period 2^96-1
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
static inline int xorshf96_fastrandombytes(void *buf, const size_t n) noexcept{
	uint64_t *a = (uint64_t *)buf;

	const uint32_t rest = n%8;
	const size_t limit = n/8;
	size_t i = 0;

	for (; i < limit; ++i) {
		a[i] = xorshf96();
	}

	// last limb
	uint8_t *b = (uint8_t *)buf;
	b += n - rest;
	uint64_t limb = xorshf96();
	for (size_t j = 0; j < rest; ++j) {
		b[j] = (limb >> (j*8u)) & 0xFFu;
	}

	return 0;
}

/// n = bytes
/// returns 0 on success
inline static int xorshf96_fastrandombytes_uint64_array(uint64_t *buf, const size_t n) noexcept {
	xorshf96_fastrandombytes(buf, n);
	return 0;
}

static inline uint64_t xorshf96_fastrandombytes_uint64() noexcept {
	return xorshf96();
	//constexpr uint32_t UINT64_POOL_SIZE = 512;    // page should be 512 * 8 Byte
	//static uint64_t tmp[UINT64_POOL_SIZE];
	//static size_t counter = 0;

	//if (counter == 0){
	//	xorshf96_fastrandombytes_uint64_array(tmp, UINT64_POOL_SIZE * 8 );
	//	counter = UINT64_POOL_SIZE;
	//}

	//counter -= 1;
	//return tmp[counter];
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
	assert(w < (sizeof(T)*8));

	T ret = (1u << w) - 1u;
	for (uint32_t i = 0; i < w; ++i) {
		const size_t to_pos = fastrandombytes_uint64() % ((sizeof(T)*8) - i);
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

/// NOTE: formally thats not uniform distribution as there is depending on the 
/// input mask, a bias. But how cares.
/// \tparam T return type
/// \tparam bits number of random bits
/// \return `bits` many random bit
template<typename T, const uint32_t bits>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
static inline T fastrandombits() noexcept {
	constexpr uint32_t Tbits = sizeof(T)*8;
	constexpr T mask = (T(1u) << bits) - T(1u);
	static_assert(Tbits >= bits);

	return T(xorshf96_fastrandombytes_uint64()) & mask;
}

#ifdef USE_AVX2
#include <immintrin.h>
///
/// \return
static __m256i fastrandombytes_m256i() noexcept {
	__m256i data;
	xorshf96_fastrandombytes_uint64_array((uint64_t *)&data, 4u * 8u);
	return data;

	//constexpr uint32_t UINT64_POOL_SIZE = 128;
	//alignas(256) static uint64_t tmp[UINT64_POOL_SIZE*4];
	//__m256i *tmp64 = (__m256i *)tmp;

	//static size_t counter = 0;

	//if (counter == 0){
	//	xorshf96_fastrandombytes_uint64_array(tmp, UINT64_POOL_SIZE * 4u * 8u);
	//	counter = UINT64_POOL_SIZE;
	//}

	//counter -= 1;
	//return tmp64[counter];
}

#endif
#endif //SMALLSECRETLWE_RANDOM_H
