/*
 Simple LZSS implementation v1.0 by Jeremy Collake
 jeremy@bitsum.com
 http://www.bitsum.com

 synopsis: 
 This is a very simple implementation of the LZSS 
 compression algorithm. I wrote it some time ago and
 have had it lying around, not going to much good use.
 So, I figured I'd make it public in hopes that someone 
 may be able to learn from it. Unfortunatly, my comments
 are non-existent.
 
 This implementation uses a 4095 (12 bit) window, with
 maximum phrase sizes of 17 bytes, and a minimum of 2 bytes,
 to allow storage within a nibble. Therefore, codewords are a 
 nice and easy 16bits. Lazy-evaluation is performed when 
 selecting the best phrase to encode. Control bits are 
 stored conveniently in groups of 8. The end of stream marker
 is indicated by a null index.

 Please remember that this code is by no means optimal, nor
 is it intended to be used for any data compression purposes.
*/

#include <algorithm>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "simd/simd.h"

#define CONTROL_LITERAL 0
#define CONTROL_CODEWORD 1

#define MAX_LENGTH 17
#define MAX_WINDOWSIZE 4095
#define ITERATIONS_BEFORE_CALLBACK 64

// TODO:
//	- class
//	- benchmark

typedef uint32_t (*PFNCOMPRESSCALLBACK)(uint8_t *, uint8_t *, uint8_t *);

/// \return number of bytes `dest` needs in `CompressData` for `src_length`
/// 	input bytes: in the worst case every byte is a literal (+ 1 control
/// 	byte per 8 tokens) plus the 2 byte end of stream marker
constexpr inline unsigned long CompressBound(const unsigned long src_length) noexcept {
	return src_length + (src_length / 8u) + 4u;
}

/// \param src[in]
/// \param out[out]
/// \return number of decompressed bytes
inline unsigned long DecompressData(const uint8_t *src,
                             uint8_t *dest) noexcept {
	uint8_t control;
	unsigned int phrase_index, control_count = 0;
	int phrase_length = 0;
	uint8_t *dest_start = dest, *temp;
	unsigned short codeword = 0;
	control = *src;
	src++;
	while (true) {
		if ((control >> 7) == CONTROL_LITERAL) {
			*dest = *src;
			dest++;
			src++;
		} else {
			codeword = *src;
			codeword <<= 8;
			codeword |= *(src + 1);
			phrase_index = codeword >> 4;
			if (!phrase_index) break;
			temp = dest - phrase_index;
			for (phrase_length = ((codeword & 0x0f) + 2);
			     phrase_length > 0;
			     phrase_length--, temp++, dest++) {
				*dest = *temp;
			}
			src += 2;
		}
		control = control << 1;
		control_count++;
		if (control_count >= 8) {
			control = *src;
			src++;
			control_count = 0;
		}
	}
	return (unsigned long) (dest - dest_start);
}

/// \param src[in]:
/// \param src[in]:
/// \param src[in]:
inline unsigned int CompressCallback(uint8_t *src,
                              uint8_t *src_end, 
                              uint8_t *p) {
	printf("\r%lu%% complete.   ",
	       ((((uintptr_t) p - (uintptr_t) src) * 100) / ((uintptr_t) src_end - (uintptr_t) src)));
	return 1;
}

/// \param str1[in]
/// \param str2[in]
/// \param maxlength[in]
/// \return 1 + the length of the common prefix of `str1` and `str2`, at most
/// 	`maxlength`. I.e. the length of a phrase whose first byte (the byte
/// 	before `str1`/`str2`) already matched.
/// NOTE: the doc said "max length on which str1==str2", which is 1 less
inline unsigned long DataCompare(const uint8_t *str1,
                          const uint8_t *str2,
                          const size_t maxlength) {
	if (maxlength == 0) { return 0; }
	unsigned long length = 1;
	// NOTE: check the length first, so nothing past `maxlength` is read
	for (; (length < maxlength) && (*str1 == *str2); length++) {
		str1++; str2++;
	}
	return length;
}

/// search for the longest chain of equal elements
/// \param str[in]:
/// \param src[in]:
/// \param maxlength[in]:
/// \param str[in]:
inline unsigned char *SearchForPhrase(uint8_t *str,
                               uint8_t *src,
                               unsigned long maxlength,
                               unsigned long *bestlength) {
	unsigned char *p = str, *best = NULL;
	unsigned long curlength;
	*bestlength = 1;
	p--;
	while (p >= src) {
		if (*p == *str) {
			// the first byte matches, so compare from the second one on
			curlength = DataCompare((p + 1), (str + 1), maxlength);
			if (curlength > *bestlength) {
				*bestlength = curlength;
				best = p;
			}
		}
		p--;
	}
	return best;
}

/// \param src[in]: input
/// \param dest[out]: output, must hold `CompressBound(src_length)` bytes
/// \param src_length[in]: number of input bytes
/// \param windowsize[in]: <= MAX_WINDOWSIZE
/// \param CallbackProc[in]: progress callback, can be nullptr
/// \return number of compressed bytes
inline unsigned long CompressData(unsigned char *src,
                           unsigned char *dest,
                           unsigned long src_length,
                           unsigned long windowsize,
                           PFNCOMPRESSCALLBACK CallbackProc) {
	assert(windowsize <= MAX_WINDOWSIZE);
	unsigned char *src_end = src + src_length,
	              *phrase_ptr = nullptr,
	              *lazy_ptr = nullptr,
	              *control_ptr,
	              *start_dest = dest,
	              *temp,
	              *p = src;
	unsigned long phrase_length = 0, maxlength, lazy_length = 0, control_counter = 0, iterationcnt = 0;
	unsigned short phrase_index;
	unsigned char control = 0;
	control_ptr = dest;
	dest++;
	// NOTE: no output bound: `dest` holds `CompressBound(src_length)` bytes,
	// stopping early would silently truncate the stream
	while (p < src_end) {
		control_counter++;
		if (control_counter == 9) {
			*control_ptr = control;
			control_ptr = dest;
			dest++;
			control = 0;
			control_counter = 1;
		}
		maxlength = std::min((unsigned long) (src_end - src),  (unsigned long) (src_end - p));
		if (maxlength > windowsize) {maxlength = windowsize;}
		if (maxlength > MAX_LENGTH) {maxlength = MAX_LENGTH;}
		temp = p - windowsize;
		if (temp < src) {
			temp = src;
		}

		if (!phrase_ptr) {
			phrase_ptr = SearchForPhrase(p, temp, maxlength, &phrase_length);
		}

		if (maxlength > 1) {
			lazy_ptr = SearchForPhrase((p + 1), (temp + 1), --maxlength, &lazy_length);
		} else {
			lazy_ptr = nullptr;
			lazy_length = 0;
		}

		if (((lazy_ptr != NULL) && lazy_length > phrase_length) || !phrase_ptr || !maxlength) {
			phrase_ptr = lazy_ptr;
			phrase_length = lazy_length;
			control <<= 1;
			control |= CONTROL_LITERAL;
			*dest = *p;
			dest++;
			p++;
		} else {
			control <<= 1;
			control |= CONTROL_CODEWORD;
			phrase_index = (unsigned short) (p - phrase_ptr);
			phrase_index = phrase_index << 4;
			phrase_index = phrase_index | (((unsigned short) phrase_length - 2) & 0x0f);
			*dest = (phrase_index >> 8);
			dest++;
			*dest = (phrase_index & 0xff);
			dest++;
			p += phrase_length;
			phrase_ptr = NULL;
		}
		iterationcnt++;
		if (iterationcnt == ITERATIONS_BEFORE_CALLBACK) {
			iterationcnt = 0;
			if (CallbackProc) {
				CallbackProc(src, src_end, p);
			}
		}
	}
	// the end of stream marker needs its own control bit
	control_counter++;
	if (control_counter == 9) {
		*control_ptr = control;
		control_ptr = dest;
		dest++;
		control = 0;
		control_counter = 1;
	}
	control <<= 1;
	control |= CONTROL_CODEWORD;
	*dest = 0;
	dest++;
	*dest = 0;
	dest++;
	while (control_counter != 8) {
		control <<= 1;
		control_counter++;
	}
	*control_ptr = control;
	if (CallbackProc) {
		CallbackProc(src, src_end, src_end);
	}
	return (unsigned long) (dest - start_dest);
}
