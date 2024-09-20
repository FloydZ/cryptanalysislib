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

/// TODO add function to compress a whole buffer


/// Source: https://arxiv.org/pdf/2403.06898
/// but, lol, there is a typo in the example code.
/// Integer compression
template<typename T>
	requires std::is_integral_v<T>
constexpr static inline size_t leb128_encode(uint8_t *buf, const T val) noexcept {
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

template<typename T>
	requires std::is_integral_v<T>
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

//template<typename T>
//	requires std::is_integral_v<T>
//constexpr static inline void leb128_bulk_decode(T *res,
//										   		const uint8_t *buf,
//												size_t n) noexcept {
//	constexpr uint32_t mask_length = 6; 
//	constexpr T mask = 0x0000808080808080;
//	uint64_t cu_val;
//	uint32_t shift_bits = 0;
//	uint32_t pt_val = 0;
//	uint32_t partial_value = 0;
//	buf -= mask_length;
//
//	// TODO goto commands instead of switch
//	while (n >= 8) {
//		buf += mask_length;
//		const uint64_t word = *(const uint64_t*)(buf);
//		const uint64_t mval = _pext_u64(word, mask);
//		switch (mval) {
//		case 0:
//		    cu_val = _pext_u64(word, 0x000000000000007f);
//		    *res++ = (cu_val << shift_bits) | pt_val;
//		    *res++ = _pext_u64(word, 0x0000000000007f00);
//		    *res++ = _pext_u64(word, 0x00000000007f0000);
//		    *res++ = _pext_u64(word, 0x000000007f000000);
//		    *res++ = _pext_u64(word, 0x0000007f00000000);
//		    *res++ = _pext_u64(word, 0x00007f0000000000);
//		    shift_bits = 0; pt_val = 0; n -= 6;
//		case 45:
//		    cu_val = _pext_u64(word, 0x0000000000007f7f);
//		    *res++ = (cu_val << shift_bits) | pt_val;
//		    *res++ = _pext_u64(word, 0x0000007f7f7f0000);
//		    pt_val = _pext_u64(word, 0x00007f0000000000);
//		    shift_bits = 7; n -= 2;
//		case 62:
//		    cu_val = _pext_u64(word, 0x000000000000007f);
//		    *res++ = (cu_val << shift_bits) | pt_val;
//		    pt_val = _pext_u64(word, 0x00007f7f7f7f7f00);
//		    shift_bits = 35; n -= 1;
//		case 63:
//		    pt_val |= _pext_u64(word, 0x00007f7f7f7f7f7f) << shift_bits;
//		    shift_bits += 42;
//		// TODOcase ... // Other cases omitted for brevity
//		}
//	}
//}
}
#endif
