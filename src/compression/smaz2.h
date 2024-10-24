#ifndef CRYPTANALYSISLIB_COMPRESSION_SMAZ2_H
#define CRYPTANALYSISLIB_COMPRESSION_SMAZ2_H

#ifndef CRYPTANALYSISLIB_COMPRESSION_H
#error "dont include this file directly. Use `#include <compression/compression.h>`"
#endif

/// original source: https://github.com/antirez/smaz2
/// Copyright (C) 2024 by Salvatore Sanfilippo -- All rights reserved.
/// This code is licensed under the MIT license. See LICENSE file for info.
/// Some changes made by FloydZ

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

/* 128 common bigrams. */
alignas(256) constexpr char *bigrams = (char *)"intherreheanonesorteattistenntartondalitseediseangoulecomeneriroderaioicliofasetvetasihamaecomceelllcaurlachhidihofonsotacnarssoprrtsassusnoiltsemctgeloeebetrnipeiepancpooldaadviunamutwimoshyoaiewowosfiepttmiopiaweagsuiddoooirspplscaywaigeirylytuulivimabty";

/* 256 common English words of length four letters or more. */
alignas(256) constexpr char *words[256] = {
        (char *)"that", (char *)"this", (char *)"with", (char *)"from", (char *)"your", (char *)"have", (char *)"more", (char *)"will", (char *)"home",
        (char *)"about", (char *)"page", (char *)"search", (char *)"free", (char *)"other", (char *)"information", (char *)"time", (char *)"they",
        (char *)"what", (char *)"which", (char *)"their", (char *)"news", (char *)"there", (char *)"only", (char *)"when", (char *)"contact", (char *)"here",
        (char *)"business", (char *)"also", (char *)"help", (char *)"view", (char *)"online", (char *)"first", (char *)"been", (char *)"would", (char *)"were",
        (char *)"some", (char *)"these", (char *)"click", (char *)"like", (char *)"service", (char *)"than", (char *)"find", (char *)"date", (char *)"back",
        (char *)"people", (char *)"list", (char *)"name", (char *)"just", (char *)"over", (char *)"year", (char *)"into", (char *)"email", (char *)"health",
        (char *)"world", (char *)"next", (char *)"used", (char *)"work", (char *)"last", (char *)"most", (char *)"music", (char *)"data", (char *)"make",
        (char *)"them", (char *)"should", (char *)"product", (char *)"post", (char *)"city", (char *)"policy", (char *)"number", (char *)"such",
        (char *)"please", (char *)"available", (char *)"copyright", (char *)"support", (char *)"message", (char *)"after", (char *)"best",
        (char *)"software", (char *)"then", (char *)"good", (char *)"video", (char *)"well", (char *)"where", (char *)"info", (char *)"right", (char *)"public",
        (char *)"high", (char *)"school", (char *)"through", (char *)"each", (char *)"order", (char *)"very", (char *)"privacy", (char *)"book", (char *)"item",
        (char *)"company", (char *)"read", (char *)"group", (char *)"need", (char *)"many", (char *)"user", (char *)"said", (char *)"does", (char *)"under",
        (char *)"general", (char *)"research", (char *)"university", (char *)"january", (char *)"mail", (char *)"full", (char *)"review",
        (char *)"program", (char *)"life", (char *)"know", (char *)"days", (char *)"management", (char *)"part", (char *)"could", (char *)"great",
        (char *)"united", (char *)"real", (char *)"international", (char *)"center", (char *)"ebay", (char *)"must", (char *)"store", (char *)"travel",
        (char *)"comment", (char *)"made", (char *)"development", (char *)"report", (char *)"detail", (char *)"line", (char *)"term", (char *)"before",
        (char *)"hotel", (char *)"send", (char *)"type", (char *)"because", (char *)"local", (char *)"those", (char *)"using", (char *)"result",
        (char *)"office", (char *)"education", (char *)"national", (char *)"design", (char *)"take", (char *)"posted", (char *)"internet",
        (char *)"address", (char *)"community", (char *)"within", (char *)"state", (char *)"area", (char *)"want", (char *)"phone", (char *)"shipping",
        (char *)"reserved", (char *)"subject", (char *)"between", (char *)"forum", (char *)"family", (char *)"long", (char *)"based", (char *)"code",
        (char *)"show", (char *)"even", (char *)"black", (char *)"check", (char *)"special", (char *)"price", (char *)"website", (char *)"index",
        (char *)"being", (char *)"women", (char *)"much", (char *)"sign", (char *)"file", (char *)"link", (char *)"open", (char *)"today", (char *)"technology",
        (char *)"south", (char *)"case", (char *)"project", (char *)"same", (char *)"version", (char *)"section", (char *)"found", (char *)"sport",
        (char *)"house", (char *)"related", (char *)"security", (char *)"both", (char *)"county", (char *)"american", (char *)"game", (char *)"member",
        (char *)"power", (char *)"while", (char *)"care", (char *)"network", (char *)"down", (char *)"computer", (char *)"system", (char *)"three",
        (char *)"total", (char *)"place", (char *)"following", (char *)"download", (char *)"without", (char *)"access", (char *)"think",
        (char *)"north", (char *)"resource", (char *)"current", (char *)"media", (char *)"control", (char *)"water", (char *)"history",
        (char *)"picture", (char *)"size", (char *)"personal", (char *)"since", (char *)"including", (char *)"guide", (char *)"shop",
        (char *)"directory", (char *)"board", (char *)"location", (char *)"change", (char *)"white", (char *)"text", (char *)"small", (char *)"rating",
        (char *)"rate", (char *)"government", (char *)"child", (char *)"during", (char *)"return", (char *)"student", (char *)"shopping",
        (char *)"account", (char *)"site", (char *)"level", (char *)"digital", (char *)"profile", (char *)"previous", (char *)"form", (char *)"event",
        (char *)"love", (char *)"main", (char *)"another", (char *)"class", (char *)"still"
};

/// Compress the string 's' of 'len' bytes and stores the compression
/// result in 'dst' for a maximum of 'dstlen' bytes. Returns the
/// amount of bytes written.
/// @param dst
/// @param dstlen
/// @param s
/// @param _len
/// @return
[[nodiscard]] constexpr unsigned long smaz2_compress(unsigned char *dst,
                             						 const size_t dstlen,
                             						 const unsigned char *s,
                             						 const size_t _len) noexcept {
	size_t len = _len;
	int verblen = 0;     /* Length of the emitted verbatim sequence, 0 if
                          * no verbating sequence was emitted last time,
                          * otherwise 1...5, it never reaches 8 even if we have
                          * vertabim len of 8, since as we emit a verbatim
                          * sequence of 8 bytes we reset verblen to 0 to
                          * star emitting a new verbatim sequence. */
	unsigned long y = 0; // Index of next byte to set in 'dst'.

	while(len && y < dstlen) {
		/* Try to emit a word. */
		if (len >= 4) {
			uint32_t i, wordlen;
			for (i = 0; i < 256; i++) {
				const char *w = words[i];
				wordlen = strlen(w);
				unsigned int space = s[0] == ' ';

				if (len >= wordlen+space &&
				    memcmp(w,s+space,wordlen) == 0) break; // Match.
			}

			/* Emit word if a match was found.
             * The escapes are:
             * byte value 6: simple word.
             * byte value 7: word + space.
             * byte value 8: space + word. */
			if (i != 256) {
				if (s[0] == ' ') {
					if (y < dstlen) dst[y++] = 8; // Space + word.
					if (y < dstlen) dst[y++] = i; // Word ID.
					s++; len--; // Account for the space.
				} else if (len > wordlen && s[wordlen] == ' ') {
					if (y < dstlen) dst[y++] = 7; // Word + space.
					if (y < dstlen) dst[y++] = i; // Word ID.
					s++; len--; // Account for the space.
				} else {
					if (y < dstlen) dst[y++] = 6; // Simple word.
					if (y < dstlen) dst[y++] = i; // Word ID.
				}

				/* Consume. */
				s += wordlen;
				len -= wordlen;
				verblen = 0;
				continue;
			}
		}

		/* Try to emit a bigram. */
		if (len >= 2) {
			int i;
			for (i = 0; i < 128; i++) {
				const char *b = bigrams + i*2;
				if (s[0] == b[0] && s[1] == b[1]) break;
			}

			/* Emit bigram if a match was found. */
			if (i != 128) {
				int x = 1;
				if (y < dstlen) dst[y++] = x<<7 | i;

				/* Consume. */
				s += 2;
				len -= 2;
				verblen = 0;
				continue;
			}
		}

		/* No word/bigram match. Let's try if we can represent this
         * byte with a single output byte without escaping. We can
         * for all the bytes values but 1, 2, 3, 4, 5, 6, 7, 8. */
		if (!(s[0] > 0 && s[0] < 9) && s[0] < 128) {
			if (y < dstlen) dst[y++] = s[0];

			/* Consume. */
			s++;
			len--;
			verblen = 0;
			continue;
		}

		/* If we are here, we got no match nor in the bigram nor
         * with the single byte. We have to emit 'varbatim' bytes
         * with the escape sequence. */
		verblen++;
		if (verblen == 1) {
			if (y+1 == dstlen) break; /* No room for 2 bytes. */
			dst[y++] = verblen;
			dst[y++] = s[0];
		} else {
			dst[y++] = s[0];
			dst[y-(verblen+1)] = verblen; // Fix the verbatim bytes length.
			if (verblen == 5) verblen = 0; // Start to emit a new sequence.
		}

		/* Consume. */
		s++;
		len--;
	}
	return y;
}

/// Decompress the string 'c' of 'len' bytes and stores the compression
/// result in 'dst' for a maximum of 'dstlen' bytes. Returns the
/// amount of bytes written.
/// \param dst
/// \param _dstlen
/// \param c
/// \param _len
/// \return
[[nodiscard]] constexpr size_t smaz2_decompress(unsigned char *dst,
                                                const size_t _dstlen,
                                                const unsigned char *c,
                                                const size_t _len) noexcept {
	unsigned long orig_dstlen = _dstlen, i = 0;

	size_t len = _len;
	size_t dstlen = _dstlen;
	while (i < len) {
		if ((c[i] & 128) != 0) {
			/* Emit bigram. */
			unsigned char idx = c[i]&127;
			if (dstlen && dstlen-- && i < len) *dst++ = bigrams[idx*2];
			if (dstlen && dstlen-- && i < len) *dst++ = bigrams[idx*2+1];
			i++;
		} else if (c[i] > 0 && c[i] < 6) {
			/* Emit verbatim sequence. */
			unsigned char vlen = c[i++];
			while(vlen-- && i < len)
				if (dstlen && dstlen--) *dst++ = c[i++];
		} else if (c[i] > 5 && c[i] < 9) {
			/* Emit word. */
			unsigned char escape = c[i];
			if (dstlen && escape == 8 && dstlen--) *dst++ = ' ';
			i++; // Go to word ID byte.
			if (i == len) return 0; // Malformed input.
			unsigned char idx = c[i++], j = 0;
			while(words[idx][j] != 0)
				if (dstlen && dstlen--) *dst++ = words[idx][j++];
			if (dstlen && escape == 7 && dstlen--) *dst++ = ' ';
		} else {
			/* Emit byte as it is. */
			if (dstlen--) *dst++ = c[i++];
		}
	}
	return orig_dstlen - dstlen;
}

/// \tparam Iterator
/// \param first
/// \param last
/// \param out
/// \return
template<class Iterator>
#if __cplusplus > 201709L
	requires std::forward_iterator<Iterator>
#endif
[[nodiscard]] constexpr Iterator smaz2_compress(Iterator &first,
												Iterator &last,
												Iterator &out) noexcept {
	const size_t size = std::distance(first, last);
	size_t outlen = size;
	outlen = smaz2_compress(&(*out), outlen, &(*first), size);
	std::advance(out, outlen);
	return out;
}

/// \tparam Iterator
/// \param first
/// \param last
/// \param out
/// \return
template<class Iterator>
#if __cplusplus > 201709L
	requires std::forward_iterator<Iterator>
#endif
[[nodiscard]] constexpr Iterator smaz2_decompress(Iterator &first,
												  Iterator &last,
												  Iterator &out) noexcept {
	const size_t size = std::distance(first, last);
	size_t outlen = size;
	outlen = smaz2_decompress(&(*out), outlen, &(*first), size);
	std::advance(out, outlen);
	return out;
}
#endif//CRYPTANALYSISLIB_SMAZ2_H
