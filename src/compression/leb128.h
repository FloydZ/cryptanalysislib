#ifndef CRYPTANALYSISLIB_COMPRESSION_INT_H
#define CRYPTANALYSISLIB_COMPRESSION_INT_H

#ifndef CRYPTANALYSISLIB_COMPRESSION_H
#error "dont include this file directly. Use `#include <compression/compression.h>`"
#endif

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "popcount/popcount.h"

namespace cryptanalysislib {

/// Source: https://arxiv.org/pdf/2403.06898
/// but, lol, there is a typo in the example code.
/// Integer compression
/// \return the size of the compressed buffer
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline size_t leb128_encode(uint8_t *buf,
                                             const T val) noexcept {
	T t = val;
	size_t ret = 0;
	while (t >= 0x80) {
		*buf = 0x80 | (t & 0x7F);
		t >>= 7;
		buf++;
		ret++;
	}
	*buf = t;
	ret += 1;
	return ret;
}

/// compress multiple elements
/// \return the size of the compressed buffer
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline size_t leb128_encode(uint8_t *buf,
                                             const T *val,
                                             const size_t n) noexcept {
    const uint8_t *tmp = buf;
    for (size_t i = 0; i < n; i++) {
        buf += leb128_encode(buf, val[i]);
    }

    return buf - tmp;
}


/// compress multiple elements
/// \return the size of the compressed buffer
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline size_t leb128_encode(std::vector<uint8_t> &buf,
                                             const std::vector<T> &val) noexcept {
    return leb128_encode(buf.data(), val.data, val.size());
}

/// integer decompression
/// \return the compressed element
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline T leb128_decode(uint8_t **buf) noexcept {
	static_assert(sizeof(T) <= 8);
	constexpr uint32_t max_shift = (sizeof(T) == 8) ? 63 :
								   (sizeof(T) == 4) ? 28 :
								   (sizeof(T) == 1) ? 14 : 7;
	T res = 0;
	for (uint32_t shift = 0; shift < max_shift; shift += 7) {
		uint8_t tmp = **buf;
		(*buf)++;
		res |= ((tmp & 0x7F) << shift);
		if (!(tmp & 0x80)) [[likely]] {
			break;
		}
	}

	return res;
}

/// integer decompression
/// \return the number of decompressed elements
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline size_t leb128_decode(T *out,
                                           const uint8_t *buf,
                                           const size_t n) noexcept {
    size_t ctr = 0;
    const uint8_t *t = buf + n;
    while (buf < t) {
        out[ctr++] = leb128_decode<T>(&buf);

    }
    return ctr; 
}

/// @param buf
/// @param n
constexpr static inline void leb128_skip(const uint8_t *buf,
										 const size_t n) noexcept {
	uint64_t *w = (uint64_t *)buf;
	size_t nn = n;
	while (nn >= 8) {
		nn -= popcount::popcount(~(*w++) & 0x8080808080808080);
	}

	buf = (uint8_t *)w;
	while(nn--) {
		while(*buf++ * 0x80) {}
	}
}

/// @param buf
/// @return
constexpr static inline size_t leb128_count(const uint8_t *buf) noexcept {
	uint64_t *w = (uint64_t *)buf;
	size_t n = 0;

	// NOTE: probably reads out of bounds.
	uint32_t k=1;
	while (k > 0) {
        k = popcount::popcount(~(*w++) & 0x8080808080808080);
		n += k;
	}

	return n;
}

} // end namespace cryptanalysislib
#endif
