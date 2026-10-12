#ifndef CRYPTANALYSISLIB_COMPRESSION_LZ77_H
#define CRYPTANALYSISLIB_COMPRESSION_LZ77_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "memory/memory.h"

#ifndef CRYPTANALYSISLIB_COMPRESSION_H
#error "dont include this file directly. Use `#include <compression/compression.h>`"
#endif

/// org code:https://github.com/andyherbert/lz1

/// LZ77 compression (greedy, brute force match search: O(n * window)).
/// Format: 4 byte uncompressed size (little endian), 1 byte
/// `pointer_length_width`, then one 3 byte token per step: a 16 bit pointer
/// (distance << pointer_length_width | (match length - 1), distance 0 = no
/// match) followed by the next literal byte.
/// \param compressed_text[out]: output buffer, at least
/// 	5 + 3 * uncompressed_size bytes (the worst case: no matches)
/// \param uncompressed_text[in]: input bytes
/// \param uncompressed_size[in]: number of input bytes (< 2^32)
/// \param pointer_length_width[in]: bits of a pointer used for the match
/// 	length (max. match 2^w), the remaining 16 - w bits are the distance
/// 	(window 2^(16-w) - 1 bytes); e.g. 4
/// \return compressed size in bytes
inline uint32_t lz77_compress(uint8_t *compressed_text,
                       const uint8_t *uncompressed_text,
                       const size_t uncompressed_size,
                       const uint8_t pointer_length_width) {
	uint16_t pointer_pos, temp_pointer_pos, output_pointer, pointer_length, temp_pointer_length;
	uint32_t compressed_pointer, output_size, coding_pos, output_lookahead_ref, look_behind, look_ahead;
	uint16_t pointer_pos_max, pointer_length_max;
	pointer_pos_max = 1u << (16 - pointer_length_width);
	pointer_length_max = 1u<< pointer_length_width;

	// NOTE: byte copies, the header and the tokens (3 bytes each) are not
	// 	aligned. Before, they were stored via `uint32_t *`/`uint16_t *`.
	const uint32_t size32 = uncompressed_size;
	cryptanalysislib::memcpy<uint8_t>(compressed_text, (const uint8_t *)&size32, 4);
	*(compressed_text + 4) = pointer_length_width;
	compressed_pointer = output_size = 5;

	for (coding_pos = 0; coding_pos < uncompressed_size; ++coding_pos) {
		pointer_pos = 0;
		pointer_length = 0;
		// NOTE: each token is a match followed by a literal, so a match may
		// 	cover at most all but the last remaining byte. Before, the match
		// 	loop had no bound and read past the end of the input.
		const uint32_t remaining = uncompressed_size - coding_pos - 1u;
		const uint32_t max_length = remaining < pointer_length_max ? remaining : pointer_length_max;
		for (temp_pointer_pos = 1; (temp_pointer_pos < pointer_pos_max) &&
		                           (temp_pointer_pos <= coding_pos);
		     ++temp_pointer_pos) {

			look_behind = coding_pos - temp_pointer_pos;
			look_ahead = coding_pos;
			for (temp_pointer_length = 0;
			     (temp_pointer_length < max_length) &&
			     (uncompressed_text[look_ahead++] == uncompressed_text[look_behind++]);
			     ++temp_pointer_length) {}
			if (temp_pointer_length > pointer_length) {
				pointer_pos = temp_pointer_pos;
				pointer_length = temp_pointer_length;
				if (pointer_length == pointer_length_max)
					break;
			}
		}

		coding_pos += pointer_length;
		if ((coding_pos == uncompressed_size) && pointer_length) {
			output_pointer = (pointer_length == 1) ? 0 : ((pointer_pos << pointer_length_width) | (pointer_length - 2));
			output_lookahead_ref = coding_pos - 1;
		} else {
			output_pointer = (pointer_pos << pointer_length_width) | (pointer_length ? (pointer_length - 1) : 0);
			output_lookahead_ref = coding_pos;
		}
		cryptanalysislib::memcpy<uint8_t>(compressed_text + compressed_pointer, (const uint8_t *)&output_pointer, 2);
		compressed_pointer += 2;
		*(compressed_text + compressed_pointer++) = *(uncompressed_text + output_lookahead_ref);
		output_size += 3;
	}

	return output_size;
}

/// LZ77 decompression of the format written by `lz77_compress`
/// \param uncompressed_text[out]: output buffer, at least the uncompressed
/// 	size (stored in the first 4 bytes of `compressed_text`)
/// \param compressed_text[in]: compressed bytes
/// \return decompressed size in bytes
inline uint32_t lz77_decompress(uint8_t *uncompressed_text,
                         const uint8_t *compressed_text) {
	uint8_t pointer_length_width;
	uint16_t input_pointer, pointer_length, pointer_pos, pointer_length_mask;
	uint32_t compressed_pointer, coding_pos, pointer_offset, uncompressed_size;

	cryptanalysislib::memcpy<uint8_t>((uint8_t *)&uncompressed_size, compressed_text, 4);
	pointer_length_width = *(compressed_text + 4);
	compressed_pointer = 5;

	pointer_length_mask = (1u << pointer_length_width) - 1;

	for (coding_pos = 0; coding_pos < uncompressed_size; ++coding_pos) {
		cryptanalysislib::memcpy<uint8_t>((uint8_t *)&input_pointer, compressed_text + compressed_pointer, 2);
		compressed_pointer += 2;
		pointer_pos = input_pointer >> pointer_length_width;
		pointer_length = pointer_pos ? ((input_pointer & pointer_length_mask) + 1) : 0;
		if (pointer_pos)
			for (pointer_offset = coding_pos - pointer_pos; pointer_length > 0; --pointer_length)
				uncompressed_text[coding_pos++] = uncompressed_text[pointer_offset++];
		*(uncompressed_text + coding_pos) = *(compressed_text + compressed_pointer++);
	}

	return coding_pos;
}

#endif
