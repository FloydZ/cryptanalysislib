#ifndef CRYPTANALYSISLIB_COMPRESSION_INT_H
#define CRYPTANALYSISLIB_COMPRESSION_INT_H

#ifndef CRYPTANALYSISLIB_COMPRESSION_H
#error "dont include this file directly. Use `#include <compression/compression.h>`"
#endif

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "algorithm/bits/popcount.h"
#include "memory/memory.h"

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
	// NOTE: the bits of `val` are encoded as unsigned, as `leb128_decode`
	// 	expects. Before, a negative `val` failed `t >= 0x80` and only its
	// 	lowest byte was written.
	using U = std::make_unsigned_t<T>;
	U t = U(val);
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
    return leb128_encode(buf.data(), val.data(), val.size());
}

/// integer decompression
/// \return the compressed element
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
constexpr static inline T leb128_decode(uint8_t **buf) noexcept {
	static_assert(sizeof(T) <= 8);
	using U = std::make_unsigned_t<T>;
	// a T needs at most ceil(bits/7) groups of 7 bits
	constexpr uint32_t bits = sizeof(T) * 8u;
	U res = 0;
	for (uint32_t shift = 0; shift < bits; shift += 7) {
		uint8_t tmp = **buf;
		(*buf)++;
		res |= U(U(tmp & 0x7F) << shift);
		if (!(tmp & 0x80)) [[likely]] {
			break;
		}
	}

	return T(res);
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
    // NOTE: the single element decoder advances a non-const pointer
    uint8_t *ptr = const_cast<uint8_t *>(buf);
    const uint8_t *t = buf + n;
    while (ptr < t) {
        out[ctr++] = leb128_decode<T>(&ptr);
    }
    return ctr; 
}

/// skips `n` compressed integers
/// NOTE: before, the result was lost (`void` and `buf` by value), i.e. the
/// 	function had no effect, and the words were read via a misaligned
/// 	`uint64_t *`. `n` was documented as number of bytes.
/// \param buf pointer to the first compressed integer
/// \param n number of integers to skip
/// \return pointer to the first byte after the `n` integers
constexpr static inline const uint8_t *leb128_skip(const uint8_t *buf,
										           const size_t n) noexcept {
	size_t nn = n;
	// each byte without the continuation bit terminates an integer. With at
	// least 8 integers left, the next 8 bytes all belong to them.
	while (nn >= 8) {
		uint64_t w;
		cryptanalysislib::memcpy<uint8_t>((uint8_t *)&w, buf, 8);
		nn -= popcount::popcount(~w & 0x8080808080808080ull);
		buf += 8;
	}

	while(nn--) {
		while(*buf++ & 0x80) {}
	}
	return buf;
}

/// NOTE: probably reads out off bounds.
/// \param buf pointer to the compressed integer
/// \return number of elements read
constexpr static inline size_t leb128_count(const uint8_t *buf) noexcept {
	auto *w = reinterpret_cast<const uint64_t *>(buf);
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
