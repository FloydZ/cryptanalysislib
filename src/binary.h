#ifndef CRYPTANALYSISLIB_BINARY_H
#define CRYPTANALYSISLIB_BINARY_H

#include <cstdint>

// TODO move to algorith
#ifdef _MSC_VER

#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)

#elif defined(__APPLE__)

// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)

#elif defined(__sun) || defined(sun)

#include <sys/byteorder.h>
#define bswap_32(x) BSWAP_32(x)
#define bswap_64(x) BSWAP_64(x)

#elif defined(__FreeBSD__)

#include <sys/endian.h>
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)

#elif defined(__OpenBSD__)

#include <sys/types.h>
#define bswap_32(x) swap32(x)
#define bswap_64(x) swap64(x)

#elif defined(__NetBSD__)

#include <sys/types.h>
#include <machine/bswap.h>
#if defined(__BSWAP_RENAME) && !defined(__bswap_32)
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#endif

#else

#include <byteswap.h>

#endif


constexpr inline static uint64_t fetch64(const char *p) noexcept {
	return ((uint64_t *)p)[0];
}

constexpr inline static uint32_t fetch32(const char *p) noexcept {
	return ((uint32_t *)p)[0];
}

template<typename T=uint32_t>
constexpr inline static T Rotate32(const T val, const int shift) noexcept {
	// Avoid shifting by 32: doing so yields an undefined result.
	return shift == T(0) ? val : ((val >> shift) | (val << ((sizeof(T)*8u) - shift)));
}


#include "algorithm/bits/popcount.h"
[[nodiscard]] constexpr static std::size_t round_up_to_power_of_two(std::size_t value) noexcept {
	if (cryptanalysislib::popcount::popcount(value) == 1u) {
		return value;
	}

	if (value == 0) {
		return 1;
	}

	--value;
	for (std::size_t i = 1; i < sizeof(std::size_t) * sizeof(uint8_t) * 8; i *= 2) {
		value |= value >> i;
	}

	return value + 1;
}
#endif//CRYPTANALYSISLIB_BINARY_H
